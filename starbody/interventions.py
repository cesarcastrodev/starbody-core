"""
Universal InterventionKnob Interface for Starbody / SMP v3.2.

Provides composable, deterministic forward-hook & weight intervention knobs:
  - GeodesicSOKKnob: exact norm-preserving SO(k) matrix-exponential rotation in U_parallel.
  - LinearPatchKnob: rank-1 additive residual steering control.
  - QuantizationStepKnob: non-destructive symmetric uniform weight + activation quantization.
  - LoRAAlphaKnob: continuous PEFT/LoRA `.scaling` modulation with exact restoration.

All `.attach(...)` context managers guarantee cleanup inside `try...finally`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from starbody.adapters import BaseModelAdapter


@dataclass
class AttachedController:
    """Live mutable control handle exposed inside an InterventionKnob context."""

    active_dose_s: Union[float, Sequence[float], torch.Tensor]
    layer_indices: List[int]
    events: List[Dict[str, Any]] = field(default_factory=list)


_VALID_POSITION_MODES = ("all", "prefill_only", "decode_only", "last_token")


class InterventionKnob(ABC):
    """Abstract intervention controller ensuring safe hook/parameter teardown."""

    @abstractmethod
    @contextlib.contextmanager
    def attach(
        self,
        adapter: BaseModelAdapter,
        dose_s: Union[float, Sequence[float], torch.Tensor],
        token_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[AttachedController]:
        """Applies intervention on context entry and restores clean state in `finally`."""


def _apply_token_mask(
    original: torch.Tensor,  # [batch, seq_len, d_model]
    steered: torch.Tensor,   # [batch, seq_len, d_model]
    token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if token_mask is None:
        return steered
    mask = token_mask.to(device=original.device, dtype=original.dtype)
    while mask.ndim < original.ndim:
        mask = mask.unsqueeze(-1)
    return original + mask * (steered - original)


class GeodesicSOKKnob(InterventionKnob):
    """
    Applies exact norm-preserving SO(k) matrix-exponential rotation R(s) = exp(s * A_skew)
    strictly inside U_parallel [d_model, rank_k], supporting both scalar doses and
    per-batch-row multi-tenant doses with pre-staged device/dtype tensor caches.
    """

    def __init__(
        self,
        u_parallel_by_layer: Dict[int, torch.Tensor],
        a_skew_by_layer: Dict[int, torch.Tensor],
        guard_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor, AttachedController], None]] = None,
        position_mode: str = "all",
        normalize_by_layers: bool = False,
    ) -> None:
        mode = str(position_mode).lower().strip()
        if mode not in _VALID_POSITION_MODES:
            raise ValueError(f"Unsupported position_mode '{position_mode}'. Choose from {_VALID_POSITION_MODES}.")
        self.u_parallel_by_layer = u_parallel_by_layer
        self.a_skew_by_layer = a_skew_by_layer
        self.guard_fn = guard_fn
        self.position_mode = mode
        self.normalize_by_layers = bool(normalize_by_layers)

    @contextlib.contextmanager
    def attach(
        self,
        adapter: BaseModelAdapter,
        dose_s: Union[float, Sequence[float], torch.Tensor],
        token_mask: Optional[torch.Tensor] = None,
        position_mode: Optional[str] = None,
    ) -> Iterator[AttachedController]:
        mode = str(position_mode or self.position_mode).lower().strip()
        if mode not in _VALID_POSITION_MODES:
            raise ValueError(f"Unsupported position_mode '{mode}'. Choose from {_VALID_POSITION_MODES}.")

        layers = adapter.get_layers()
        target_indices = sorted(idx for idx in self.u_parallel_by_layer if 0 <= idx < len(layers))
        if not target_indices:
            raise ValueError("No matching layer indices between GeodesicSOKKnob and adapter.")

        layer_scale = (1.0 / len(target_indices)) if (self.normalize_by_layers and len(target_indices) > 1) else 1.0

        init_dose: Union[float, Sequence[float], torch.Tensor] = (
            float(dose_s) if isinstance(dose_s, (int, float)) else dose_s
        )
        ctrl = AttachedController(active_dose_s=init_dose, layer_indices=target_indices)
        handles: List[Any] = []

        # Pre-staged caches per (layer_idx, device, dtype) to eliminate hot-path `.to()` copies
        u_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        a64_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        rot_cache: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}

        def _get_u(l_idx: int, dev: torch.device, out_dtype: torch.dtype) -> torch.Tensor:
            key = (l_idx, dev, out_dtype)
            if key not in u_cache:
                u_cache[key] = self.u_parallel_by_layer[l_idx].detach().to(device=dev, dtype=out_dtype).contiguous()
            return u_cache[key]

        def _get_a64(l_idx: int, dev: torch.device) -> torch.Tensor:
            key = (l_idx, dev)
            if key not in a64_cache:
                a64_cache[key] = self.a_skew_by_layer[l_idx].detach().to(device=dev, dtype=torch.float64).contiguous()
            return a64_cache[key]

        def _get_scalar_rotation(l_idx: int, s_val: float, dev: torch.device, out_dtype: torch.dtype) -> torch.Tensor:
            eff_s = float(s_val) * layer_scale
            key = (l_idx, eff_s, dev, out_dtype)
            if key not in rot_cache:
                a_64 = _get_a64(l_idx, dev)  # [rank_k, rank_k]
                rot_64 = torch.matrix_exp(eff_s * a_64)  # [rank_k, rank_k]
                rot_cache[key] = rot_64.to(dtype=out_dtype).contiguous()
            return rot_cache[key]

        def _rotate_slice(
            x_slice: torch.Tensor,  # [batch, seq_slice, d_model] or [batch, d_model]
            u_mat: torch.Tensor,    # [d_model, rank_k]
            l_idx: int,
            cur_s: Union[float, Sequence[float], torch.Tensor],
        ) -> torch.Tensor:
            is_scalar = isinstance(cur_s, (int, float)) or (isinstance(cur_s, torch.Tensor) and cur_s.ndim == 0)
            if is_scalar:
                s_float = float(cur_s.item()) if isinstance(cur_s, torch.Tensor) else float(cur_s)
                if abs(s_float) <= 1e-12:
                    return x_slice
                rot = _get_scalar_rotation(l_idx, s_float, x_slice.device, x_slice.dtype)  # [rank_k, rank_k]
                coeffs = x_slice @ u_mat  # [batch, ..., rank_k]
                rot_coeffs = coeffs @ rot.T  # [batch, ..., rank_k]
                return x_slice + (rot_coeffs - coeffs) @ u_mat.T

            doses_64 = torch.as_tensor(cur_s, device=x_slice.device, dtype=torch.float64).reshape(-1)  # [batch]
            if doses_64.shape[0] != x_slice.shape[0]:
                raise ValueError(
                    f"Per-row dose length {doses_64.shape[0]} does not match batch size {x_slice.shape[0]}"
                )
            active_rows = torch.abs(doses_64) > 1e-12
            if not bool(torch.any(active_rows).item()):
                return x_slice

            a_64 = _get_a64(l_idx, x_slice.device)  # [rank_k, rank_k]
            rot_64 = torch.matrix_exp((doses_64 * layer_scale).view(-1, 1, 1) * a_64.unsqueeze(0))  # [batch, rank_k, rank_k]
            rot_b = rot_64.to(dtype=x_slice.dtype)  # [batch, rank_k, rank_k]
            coeffs = x_slice @ u_mat  # [batch, seq_slice, rank_k] or [batch, rank_k]
            if x_slice.ndim == 2:
                rot_coeffs = torch.bmm(coeffs.unsqueeze(1), rot_b.transpose(1, 2)).squeeze(1)
                row_mask = active_rows.to(dtype=x_slice.dtype).view(-1, 1)
            else:
                rot_coeffs = torch.bmm(coeffs, rot_b.transpose(1, 2))
                row_mask = active_rows.to(dtype=x_slice.dtype).view(-1, 1, 1)
            delta_orth = ((rot_coeffs - coeffs) @ u_mat.T) * row_mask
            return x_slice + delta_orth

        def _make_hook(l_idx: int) -> Callable[[nn.Module, Any, Any], Any]:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                def _transform(x: torch.Tensor) -> torch.Tensor:
                    # x shape: [batch, seq_len, d_model]
                    if mode == "prefill_only" and (x.ndim < 3 or x.shape[1] <= 1):
                        return x
                    if mode == "decode_only" and x.ndim >= 3 and x.shape[1] > 1:
                        return x

                    u_mat = _get_u(l_idx, x.device, x.dtype)  # [d_model, rank_k]
                    if self.guard_fn is not None:
                        self.guard_fn(l_idx, x, u_mat, ctrl)
                    cur_s = ctrl.active_dose_s

                    if mode == "last_token" and x.ndim >= 3 and x.shape[1] > 1:
                        last_steered = _rotate_slice(x[:, -1:, :], u_mat, l_idx, cur_s)
                        steered = x.clone()
                        steered[:, -1:, :] = last_steered
                    else:
                        steered = _rotate_slice(x, u_mat, l_idx, cur_s)

                    return _apply_token_mask(x, steered, token_mask)

                return adapter.normalize_hook_output(out, _transform)

            return _hook

        try:
            for l_idx in target_indices:
                handles.append(layers[l_idx].register_forward_hook(_make_hook(l_idx)))
            yield ctrl
        finally:
            for h in handles:
                h.remove()


class LinearPatchKnob(InterventionKnob):
    """Additive rank-1 residual steering control for head-to-head selectivity comparisons."""

    def __init__(self, direction_by_layer: Dict[int, torch.Tensor]) -> None:
        self.direction_by_layer: Dict[int, torch.Tensor] = {}
        for l_idx, vec in direction_by_layer.items():
            v64 = vec.detach().to(torch.float64).reshape(-1)
            norm = torch.norm(v64).clamp(min=1e-12)
            self.direction_by_layer[l_idx] = (v64 / norm).to(torch.float32)  # [d_model]

    @contextlib.contextmanager
    def attach(
        self,
        adapter: BaseModelAdapter,
        dose_s: float,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[AttachedController]:
        layers = adapter.get_layers()
        target_indices = sorted(idx for idx in self.direction_by_layer if 0 <= idx < len(layers))
        ctrl = AttachedController(active_dose_s=float(dose_s), layer_indices=target_indices)
        handles: List[Any] = []

        def _make_hook(l_idx: int) -> Callable[[nn.Module, Any, Any], Any]:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                def _transform(x: torch.Tensor) -> torch.Tensor:
                    dir_v = self.direction_by_layer[l_idx].to(device=x.device, dtype=x.dtype)  # [d_model]
                    steered = x + ctrl.active_dose_s * dir_v
                    return _apply_token_mask(x, steered, token_mask)

                return adapter.normalize_hook_output(out, _transform)

            return _hook

        try:
            for l_idx in target_indices:
                handles.append(layers[l_idx].register_forward_hook(_make_hook(l_idx)))
            yield ctrl
        finally:
            for h in handles:
                h.remove()


class QuantizationStepKnob(InterventionKnob):
    """
    Continuous symmetric uniform weight & activation quantization step Delta = dose_s.
    Saves pristine weights before entering context and guarantees exact restoration in `finally`.
    """

    def __init__(self, target_layers: Optional[Sequence[int]] = None) -> None:
        self.target_layers = list(target_layers) if target_layers is not None else None

    @contextlib.contextmanager
    def attach(
        self,
        adapter: BaseModelAdapter,
        dose_s: float,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[AttachedController]:
        layers = adapter.get_layers()
        target_indices = (
            [i for i in self.target_layers if 0 <= i < len(layers)]
            if self.target_layers is not None
            else list(range(len(layers)))
        )
        ctrl = AttachedController(active_dose_s=float(dose_s), layer_indices=target_indices)
        saved_weights: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        handles: List[Any] = []

        delta = float(dose_s)

        def _quantize_tensor(t: torch.Tensor, step: float) -> torch.Tensor:
            if step <= 1e-12:
                return t
            return step * torch.round(t / step)

        def _make_hook() -> Callable[[nn.Module, Any, Any], Any]:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                def _transform(x: torch.Tensor) -> torch.Tensor:
                    steered = _quantize_tensor(x, ctrl.active_dose_s)
                    return _apply_token_mask(x, steered, token_mask)

                return adapter.normalize_hook_output(out, _transform)

            return _hook

        try:
            with torch.no_grad():
                for l_idx in target_indices:
                    layer = layers[l_idx]
                    for p in layer.parameters():
                        if p.ndim >= 2:
                            saved_weights.append((p, p.detach().clone()))
                            if delta > 1e-12:
                                p.copy_(_quantize_tensor(p, delta))
                    handles.append(layer.register_forward_hook(_make_hook()))
            yield ctrl
        finally:
            for h in handles:
                h.remove()
            with torch.no_grad():
                for param, orig in saved_weights:
                    param.copy_(orig)


class LoRAAlphaKnob(InterventionKnob):
    """
    Dynamically scales LoRA `.scaling` attributes on adapted modules by `dose_s`,
    restoring exact baseline scaling values on context exit inside `try...finally`.
    """

    def __init__(self, peft_model: Optional[nn.Module] = None) -> None:
        self.peft_model = peft_model

    @contextlib.contextmanager
    def attach(
        self,
        adapter: BaseModelAdapter,
        dose_s: float,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[AttachedController]:
        target_root = self.peft_model if self.peft_model is not None else adapter.model
        if target_root is None:
            raise ValueError("LoRAAlphaKnob requires a model or peft_model.")

        saved_scalings: List[Tuple[nn.Module, Any]] = []
        ctrl = AttachedController(active_dose_s=float(dose_s), layer_indices=list(range(adapter.num_layers)))

        try:
            for mod in target_root.modules():
                if hasattr(mod, "scaling"):
                    orig_val = getattr(mod, "scaling")
                    if isinstance(orig_val, dict):
                        saved_scalings.append((mod, dict(orig_val)))
                        mod.scaling = {k: float(v) * float(dose_s) for k, v in orig_val.items()}
                    elif isinstance(orig_val, (int, float)):
                        saved_scalings.append((mod, float(orig_val)))
                        mod.scaling = float(orig_val) * float(dose_s)

            if not saved_scalings:
                raise ValueError(
                    "No LoRA layers exposing `.scaling` found on target model."
                )
            yield ctrl
        finally:
            for mod, orig_val in saved_scalings:
                setattr(mod, "scaling", orig_val)

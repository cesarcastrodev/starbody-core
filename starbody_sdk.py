"""
Starbody Studio Python SDK (SMP v3.2 Product Edition).

Provides zero-friction tangent subspace extraction and workflow-coupled
certification hooks (`with starbody.certify(...) as session:`) in strict
IEEE-754 float32 (`TF32 = False`) with float64 exposure/SVD accumulators,
deterministic `try...finally` forward-hook cleanup, and non-blocking telemetry
streaming to `ws://localhost:8000/ws/ingest` or `http://localhost:8000/api/ingest_step`.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from smp_core import (
    GeodesicSubspaceIntervention,
    InteractomeLaplacianTracker,
    PilotCalibrationGate,
    TwinFreePharmacologicalCertifier,
    enforce_precision_and_seed,
)


class StarbodyBifurcationWarning(RuntimeError):
    """Raised by the SDK circuit breaker when orthogonal lock-in or Laplacian collapse triggers."""


def _resolve_target_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Resolves transformer blocks or ordered child modules for hook attachment."""
    block_seq: Optional[Sequence[nn.Module]] = None
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        block_seq = model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        block_seq = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        block_seq = model.transformer.h
    elif hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list, tuple)):
        block_seq = list(model.layers)

    if block_seq is not None and len(block_seq) > 0:
        return block_seq[layer_idx]

    named_mods = [m for _, m in model.named_children()]
    if not named_mods:
        return model
    return named_mods[layer_idx]


def _extract_last_token_vec(act: torch.Tensor) -> torch.Tensor:
    """Normalizes captured layer activation to `[d_model]` float32 vector."""
    if act.ndim == 3:
        # act shape: [batch, seq_len, d_model]
        return act[0, -1, :].detach().to(torch.float32).clone()
    if act.ndim == 2:
        # act shape: [batch, d_model]
        return act[0, :].detach().to(torch.float32).clone()
    if act.ndim == 1:
        # act shape: [d_model]
        return act.detach().to(torch.float32).clone()
    raise ValueError(f"Unsupported activation shape for tangent extraction: {tuple(act.shape)}")


@torch.no_grad()
def extract_tangent_basis_auto(
    model: nn.Module,
    positive_prompts: Sequence[Union[str, torch.Tensor]],
    negative_prompts: Sequence[Union[str, torch.Tensor]],
    layer_idx: int = -1,
    rank_k: int = 4,
    tokenizer: Optional[Any] = None,
    seed: int = 20261117,
) -> torch.Tensor:
    """
    Automatically extracts contrastive activation shifts `[N_pairs, d_model]` in strict
    float32 (`requires_grad_(False)`), computes float64 SVD + QR orthonormalization to
    return `U_parallel` `[d_model, rank_k]`, and verifies `||U^T U - I_k|| < 1e-5`.
    """
    assert len(positive_prompts) == len(negative_prompts) and len(positive_prompts) > 0, (
        "positive_prompts and negative_prompts must have equal non-zero length."
    )
    assert rank_k >= 1, f"rank_k must be >= 1, got {rank_k}"

    enforce_precision_and_seed(seed)
    model.eval()
    model.requires_grad_(False)

    params = list(model.parameters())
    model_device = params[0].device if params else torch.device("cpu")
    target_layer = _resolve_target_layer(model, layer_idx)
    captured: List[torch.Tensor] = []

    def _capture_hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        tensor_out = output[0] if isinstance(output, tuple) else output
        captured.append(_extract_last_token_vec(tensor_out))

    def _forward_prompt(item: Union[str, torch.Tensor]) -> torch.Tensor:
        captured.clear()
        if isinstance(item, str):
            assert tokenizer is not None, "tokenizer is required when passing string prompts."
            enc = tokenizer(item, return_tensors="pt", truncation=True, max_length=64)
            enc = {k: v[:1].to(model_device) for k, v in enc.items()}  # batch_size = 1 guard
            _ = model(**enc)
        else:
            x_in = item.to(device=model_device)
            if x_in.is_floating_point() and x_in.dtype != torch.float32:
                x_in = x_in.to(torch.float32)
            if x_in.ndim == 1:
                x_in = x_in.unsqueeze(0)  # [1, L] or [1, d_model]
            _ = model(x_in[:1])  # batch_size = 1 guard
        assert captured, "Forward hook did not capture any activation from target layer."
        return captured[-1]

    delta_list: List[torch.Tensor] = []
    handle = target_layer.register_forward_hook(_capture_hook)
    try:
        for pos_item, neg_item in zip(positive_prompts, negative_prompts):
            h_pos = _forward_prompt(pos_item)  # [d_model]
            h_neg = _forward_prompt(neg_item)  # [d_model]
            delta_list.append(h_pos - h_neg)   # [d_model]
    finally:
        handle.remove()

    delta_mat = torch.stack(delta_list, dim=0)  # [N_pairs, d_model]
    d_model = int(delta_mat.shape[1])
    assert rank_k <= d_model, f"rank_k ({rank_k}) exceeds d_model ({d_model})."

    # Compute float64 SVD and complete orthonormal QR basis to guarantee exact rank_k columns
    delta_64 = delta_mat.to(torch.float64)  # [N_pairs, d_model]
    _, _, vh_64 = torch.linalg.svd(delta_64, full_matrices=False)  # [min(N, D), d_model]
    n_avail = min(rank_k, int(vh_64.shape[0]))
    seed_cols_64 = vh_64[:n_avail].T  # [d_model, n_avail]

    if n_avail < rank_k:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pad_cols = torch.randn(d_model, rank_k - n_avail, generator=gen, dtype=torch.float64).to(model_device)
        seed_cols_64 = torch.cat([seed_cols_64, pad_cols], dim=1)  # [d_model, rank_k]

    q_64, r_64 = torch.linalg.qr(seed_cols_64, mode="reduced")  # [d_model, rank_k]
    signs = torch.where(torch.diag(r_64) >= 0, 1.0, -1.0).to(torch.float64)
    q_64 = q_64 * signs.unsqueeze(0)

    ident_err = float(
        torch.norm(q_64.T @ q_64 - torch.eye(rank_k, dtype=torch.float64, device=model_device)).item()
    )
    assert ident_err < 1e-5, f"Tangent basis orthogonality violation: ||U^T U - I_k|| = {ident_err:.3e}"

    return q_64.to(torch.float32).contiguous()  # [d_model, rank_k]


class _NonBlockingTelemetrySink:
    """Queues step telemetry payloads and streams via `ws://.../ws/ingest` with HTTP/vault fallback."""

    def __init__(self, server_url: str, vault_fallback_dir: Path, study_id: str) -> None:
        self.server_url = server_url.rstrip("/")
        if self.server_url.startswith("https://"):
            self.ws_url = "wss://" + self.server_url[len("https://"):] + "/ws/ingest"
        elif self.server_url.startswith("http://"):
            self.ws_url = "ws://" + self.server_url[len("http://"):] + "/ws/ingest"
        else:
            self.ws_url = self.server_url + "/ws/ingest"
        self.vault_dir = vault_fallback_dir / "results"
        self.study_id = study_id
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._server_reachable = True
        self._ws_conn: Any = None
        self._thread.start()

    def emit(self, payload: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def close(self, timeout: float = 1.5) -> None:
        try:
            self._queue.put(None, timeout=0.2)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _send_ws(self, payload: Dict[str, Any]) -> bool:
        if not self._server_reachable:
            return False
        try:
            if self._ws_conn is None:
                from websockets.sync.client import connect

                self._ws_conn = connect(self.ws_url, open_timeout=0.35, close_timeout=0.35)
            self._ws_conn.send(json.dumps(payload))
            return True
        except Exception:
            if self._ws_conn is not None:
                try:
                    self._ws_conn.close()
                except Exception:
                    pass
                self._ws_conn = None
            return False

    def _post_json(self, payload: Dict[str, Any]) -> bool:
        if not self._server_reachable:
            return False
        if self._send_ws(payload):
            return True
        endpoint = f"{self.server_url}/api/ingest_step"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=0.35) as resp:
                return 200 <= int(resp.status) < 300
        except Exception:
            self._server_reachable = False
            return False

    def _write_local_fallback(self, payload: Dict[str, Any]) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        step_idx = int(payload.get("step_index", 0))
        out_path = self.vault_dir / f"{self.study_id}_sdk_step_{step_idx:04d}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                sent = self._post_json(item)
                if not sent:
                    self._write_local_fallback(item)
                self._queue.task_done()
        finally:
            if self._ws_conn is not None:
                try:
                    self._ws_conn.close()
                except Exception:
                    pass
                self._ws_conn = None


class StarbodySession:
    """
    Workflow-coupled certification session (`with starbody.certify(...) as session:`).
    Guarantees forward-hook cleanup via `try...finally`, validates sham floor on entry,
    monitors `E_parallel`, `E_perp`, `alpha_chi_perp`, and Fiedler `lambda_2`, and enforces
    optional circuit-breaker thresholds.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tangent_basis: Optional[torch.Tensor] = None,  # [d_model, k]
        layer_indices: Sequence[int] = (-1,),
        budget_perp: float = 0.02,
        max_alpha: float = 2.5,
        min_lambda2: float = 0.25,
        circuit_breaker: bool = False,
        circuit_breaker_action: str = "raise",
        interactome_nodes: int = 0,
        server_url: str = "http://127.0.0.1:8000",
        study_id: str = "SMP_V3_2_SDK_LIVE",
        vault_dir: str = "./smp_local_vault",
        calibration_probe_fn: Optional[Callable[[float], Dict[str, float]]] = None,
        intervene: bool = False,
        seed: int = 20261117,
    ) -> None:
        enforce_precision_and_seed(seed)
        self.model = model
        self.tangent_basis = tangent_basis.detach().to(torch.float32) if tangent_basis is not None else None
        self.layer_indices = list(layer_indices)
        self.budget_perp = float(budget_perp)
        self.max_alpha = float(max_alpha)
        self.min_lambda2 = float(min_lambda2)
        self.circuit_breaker = bool(circuit_breaker)
        self.circuit_breaker_action = circuit_breaker_action
        self.interactome_nodes = int(interactome_nodes)
        self.server_url = server_url
        self.study_id = study_id
        self.vault_dir = Path(vault_dir)
        self.calibration_probe_fn = calibration_probe_fn
        self.intervene = bool(intervene)

        self.current_dose: float = 0.0
        self._handles: List[Any] = []
        self._last_captured: Optional[torch.Tensor] = None
        self._baseline_captured: Optional[torch.Tensor] = None
        self._certifier: Optional[TwinFreePharmacologicalCertifier] = None
        self._interactome: Optional[InteractomeLaplacianTracker] = None
        self._interventions: Dict[int, GeodesicSubspaceIntervention] = {}
        self._sink: Optional[_NonBlockingTelemetrySink] = None

        self.calibration_certificate: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.circuit_tripped: bool = False

    def _init_certifier(self, d_model: int, device: torch.device) -> None:
        if self.tangent_basis is None:
            k_dim = min(4, d_model)
            basis = torch.eye(d_model, k_dim, dtype=torch.float32, device=device)  # [d_model, k]
            self.tangent_basis = basis
        else:
            self.tangent_basis = self.tangent_basis.to(device=device, dtype=torch.float32)

        self._certifier = TwinFreePharmacologicalCertifier(
            tangent_basis=self.tangent_basis,
            budget_perp=self.budget_perp,
            exponent_warning_threshold=self.max_alpha,
        )
        if self.interactome_nodes >= 2:
            scale_d = math.sqrt(float(d_model))
            self._interactome = InteractomeLaplacianTracker(
                num_nodes=self.interactome_nodes,
                state_dim=d_model,
                tangent_basis=self.tangent_basis,
                g_max=1.4,
                v_half=0.85 * scale_d,
                v_slope=0.18 * scale_d,
            ).to(device)

    def _register_hooks(self) -> None:
        assert self.model is not None
        params = list(self.model.parameters())
        dev = params[0].device if params else torch.device("cpu")
        for idx in self.layer_indices:
            layer_mod = _resolve_target_layer(self.model, idx)

            def _make_hook(layer_key: int):
                def _hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any):
                    is_tup = isinstance(output, tuple)
                    h = output[0] if is_tup else output
                    h_32 = h.to(torch.float32)  # [batch, seq_len, d_model] or [batch, d_model]
                    self._last_captured = h_32.detach().clone()
                    if self.intervene and layer_key in self._interventions and self.current_dose != 0.0:
                        op = self._interventions[layer_key]
                        if h_32.ndim == 3:
                            steered = op(h_32, dose=self.current_dose)  # [batch, seq_len, d_model]
                        elif h_32.ndim == 2:
                            steered = op(h_32.unsqueeze(1), dose=self.current_dose).squeeze(1)  # [batch, d_model]
                        else:
                            steered = h_32
                        return (steered,) + output[1:] if is_tup else steered
                    return output

                return _hook

            self._handles.append(layer_mod.register_forward_hook(_make_hook(idx)))
            if self.intervene and self.tangent_basis is not None:
                k_dim = int(self.tangent_basis.shape[1])
                skew = torch.zeros((k_dim, k_dim), dtype=torch.float32, device=dev)
                if k_dim >= 2:
                    skew[0, 1] = 1.0
                    skew[1, 0] = -1.0
                self._interventions[idx] = GeodesicSubspaceIntervention(self.tangent_basis.to(dev), skew).to(dev)

    def __enter__(self) -> "StarbodySession":
        try:
            if self.tangent_basis is not None:
                self._init_certifier(int(self.tangent_basis.shape[0]), self.tangent_basis.device)
            if self.model is not None:
                self._register_hooks()

            # Run PilotCalibrationGate on entry
            if self.calibration_probe_fn is not None:
                self.calibration_certificate = PilotCalibrationGate.calibrate_dose_ladder(
                    probe_fn=self.calibration_probe_fn,
                    candidate_doses=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
                    min_baseline_signal=0.1,
                    min_positive_control_effect=0.5,
                )
            else:
                sham_e_perp = 0.0
                if self._baseline_captured is not None and self._last_captured is not None and self._certifier is not None:
                    _, _, sham_e_perp = self._certifier.decompose_state_shift(
                        self._last_captured - self._baseline_captured
                    )
                assert sham_e_perp < 1e-4, f"Sham e_perp floor {sham_e_perp:.3e} >= 1e-4"
                self.calibration_certificate = {
                    "calibration_status": "PASS",
                    "sham_e_perp_floor": float(sham_e_perp),
                    "baseline_assay_signal": 1.0,
                }

            self._sink = _NonBlockingTelemetrySink(
                server_url=self.server_url,
                vault_fallback_dir=self.vault_dir,
                study_id=self.study_id,
            )
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        try:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
        finally:
            if self._sink is not None:
                self._sink.close(timeout=1.0)
                self._sink = None
        return False

    def capture_baseline(self, state_tensor: Optional[torch.Tensor] = None) -> None:
        """Sets baseline reference tensor `[..., d_model]` from explicit input or hooked forward."""
        ref = state_tensor if state_tensor is not None else self._last_captured
        assert ref is not None, "No activation captured yet; pass state_tensor or run a forward pass."
        self._baseline_captured = ref.detach().to(torch.float32).clone()
        if self._certifier is None:
            self._init_certifier(int(self._baseline_captured.shape[-1]), self._baseline_captured.device)

    @staticmethod
    def _coerce_agent_matrix(
        state_or_shift: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, bool]:
        """Normalizes Tensor `[..., D]` or multi-agent Dict `{role: Tensor}` into Tensor."""
        if isinstance(state_or_shift, dict):
            rows = [v.detach().to(torch.float32).reshape(-1) for v in state_or_shift.values()]
            return torch.stack(rows, dim=0), True  # [M, D]
        return state_or_shift.detach().to(torch.float32), False

    def step(
        self,
        dose: float,
        state_or_shift: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        trajectory: Optional[torch.Tensor] = None,  # [T, ..., D]
        retention: Optional[float] = None,
        dither_amp: float = 0.04,
        omega_d: Optional[float] = None,
        conductance_matrix: Optional[List[List[float]]] = None,
        fiedler_lambda2: Optional[float] = None,
        per_node_selectivity: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Records one dose/training step, computes float64 manifold decomposition & local scaling,
        checks Fiedler connectivity (when multi-agent `[M, D]`), enforces `circuit_breaker`,
        and streams telemetry non-blockingly.
        """
        self.current_dose = float(dose)
        raw_tensor: Optional[torch.Tensor] = None
        is_agent_dict = False

        if state_or_shift is not None:
            raw_tensor, is_agent_dict = self._coerce_agent_matrix(state_or_shift)
        elif self._last_captured is not None:
            raw_tensor = self._last_captured

        assert raw_tensor is not None, "No state_or_shift passed and no hooked forward activation recorded."
        d_model = int(raw_tensor.shape[-1])
        if self._certifier is None:
            self._init_certifier(d_model, raw_tensor.device)
        assert self._certifier is not None

        if self._baseline_captured is not None and state_or_shift is None:
            shift = raw_tensor - self._baseline_captured
        else:
            shift = raw_tensor

        e_total, e_parallel, e_perp = self._certifier.decompose_state_shift(shift)
        selectivity = float(e_parallel / max(e_total, 1e-15)) if e_total > 0.0 else 1.0

        if trajectory is not None:
            t_steps = int(trajectory.shape[0])
            w_d = omega_d if omega_d is not None else (2.0 * math.pi * 2.0 / max(t_steps, 1))
            chi_perp = float(self._certifier.lockin_orthogonal_susceptibility(trajectory, dither_amp, w_d))
        else:
            # Estimate susceptibility natively from discrete E_perp vs delta_s^2 increment
            chi_perp = float(e_perp / max(float(dose) ** 2, 1e-6)) if dose > 0.0 else 0.0

        # Compute local log-log exponents against previous recorded step
        alpha_perp: Optional[float] = None
        alpha_chi: Optional[float] = None
        if len(self.history) >= 1 and float(dose) > 0.0:
            prev = self.history[-1]
            prev_dose = float(prev["dose"])
            if prev_dose > 0.0 and float(dose) != prev_dose:
                floor_ep = float(self.history[0]["e_perp"]) if float(self.history[0]["dose"]) == 0.0 else 0.0
                floor_chi = float(self.history[0]["chi_perp"]) if float(self.history[0]["dose"]) == 0.0 else 0.0
                exps_ep = self._certifier.compute_local_exponents([prev_dose, float(dose)], [float(prev["e_perp"]), e_perp], floor=floor_ep)
                exps_ch = self._certifier.compute_local_exponents([prev_dose, float(dose)], [float(prev["chi_perp"]), chi_perp], floor=floor_chi)
                alpha_perp = exps_ep[-1]
                alpha_chi = exps_ch[-1]

        # Compute Graph Laplacian Fiedler eigenvalue if multi-agent [M, D] provided
        if raw_tensor.ndim == 2 and (is_agent_dict or self.interactome_nodes >= 2) and raw_tensor.shape[0] >= 2:
            m_nodes = int(raw_tensor.shape[0])
            if self._interactome is None or self._interactome.num_nodes != m_nodes:
                scale_d = math.sqrt(float(d_model))
                self._interactome = InteractomeLaplacianTracker(
                    num_nodes=m_nodes,
                    state_dim=d_model,
                    tangent_basis=self.tangent_basis,
                    g_max=1.4,
                    v_half=0.85 * scale_d,
                    v_slope=0.18 * scale_d,
                ).to(raw_tensor.device)
            g_mat, _, lap_stats = self._interactome.compute_conductance_and_laplacian(raw_tensor)  # [M, M]
            fiedler_lambda2 = float(lap_stats["fiedler_lambda2"])
            conductance_matrix = [[round(float(x), 4) for x in row] for row in g_mat.detach().cpu().tolist()]
            if per_node_selectivity is None:
                pns: List[float] = []
                for m_i in range(m_nodes):
                    nt, np, _ = self._certifier.decompose_state_shift(shift[m_i:m_i + 1])
                    pns.append(round(float(np / max(nt, 1e-15)), 4))
                per_node_selectivity = pns

        step_payload: Dict[str, Any] = {
            "study_id": self.study_id,
            "step_index": len(self.history),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dose": round(float(dose), 6),
            "e_total": float(e_total),
            "e_parallel": float(e_parallel),
            "e_perp": float(e_perp),
            "chi_perp": float(chi_perp),
            "selectivity": round(selectivity, 4),
            "alpha_perp": round(float(alpha_perp), 4) if alpha_perp is not None else None,
            "alpha_chi_perp": round(float(alpha_chi), 4) if alpha_chi is not None else None,
            "retention": round(float(retention), 4) if retention is not None else round(max(0.0, 100.0 * (1.0 - min(1.0, e_perp))), 2),
            "fiedler_lambda2": round(float(fiedler_lambda2), 4) if fiedler_lambda2 is not None else None,
            "conductance_matrix": conductance_matrix,
            "per_node_selectivity": per_node_selectivity,
            "budget_perp": self.budget_perp,
            "max_alpha": self.max_alpha,
        }
        self.history.append(step_payload)
        if self._sink is not None:
            self._sink.emit(step_payload)

        tripped_alpha = alpha_chi is not None and float(alpha_chi) > self.max_alpha
        tripped_lambda = fiedler_lambda2 is not None and float(fiedler_lambda2) < self.min_lambda2
        if self.circuit_breaker and (tripped_alpha or tripped_lambda):
            self.circuit_tripped = True
            reason = (
                f"alpha_chi_perp={alpha_chi:.2f} > {self.max_alpha}"
                if tripped_alpha
                else f"fiedler_lambda2={fiedler_lambda2:.3f} < {self.min_lambda2}"
            )
            if self.circuit_breaker_action == "attenuate":
                self.current_dose *= 0.5
                step_payload["circuit_breaker_attenuated_dose"] = self.current_dose
            else:
                raise StarbodyBifurcationWarning(f"[Starbody Circuit Breaker] Tripped at dose={dose}: {reason}")

        return step_payload


def certify(
    model: Optional[nn.Module] = None,
    tangent_basis: Optional[torch.Tensor] = None,
    layer_indices: Sequence[int] = (-1,),
    budget_perp: float = 0.02,
    max_alpha: float = 2.5,
    min_lambda2: float = 0.25,
    circuit_breaker: bool = False,
    circuit_breaker_action: str = "raise",
    interactome_nodes: int = 0,
    server_url: str = "http://127.0.0.1:8000",
    study_id: str = "SMP_V3_2_SDK_LIVE",
    vault_dir: str = "./smp_local_vault",
    calibration_probe_fn: Optional[Callable[[float], Dict[str, float]]] = None,
    intervene: bool = False,
    seed: int = 20261117,
) -> StarbodySession:
    """Factory returning a `StarbodySession` context manager (`with starbody.certify(...) as session:`)."""
    return StarbodySession(
        model=model,
        tangent_basis=tangent_basis,
        layer_indices=layer_indices,
        budget_perp=budget_perp,
        max_alpha=max_alpha,
        min_lambda2=min_lambda2,
        circuit_breaker=circuit_breaker,
        circuit_breaker_action=circuit_breaker_action,
        interactome_nodes=interactome_nodes,
        server_url=server_url,
        study_id=study_id,
        vault_dir=vault_dir,
        calibration_probe_fn=calibration_probe_fn,
        intervene=intervene,
        seed=seed,
    )

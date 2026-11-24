"""
3-Line Developer Onboarding & Bidirectional Starbody Studio Worker Client.

Usage:
    import starbody
    conn = starbody.connect("HuggingFaceTB/SmolLM2-135M")
    conn.sweep_ladder()
    conn.compile("my_model.starbody")
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from smp_core import (
    TwinFreePharmacologicalCertifier,
    enforce_precision_and_seed,
)
from starbody.adapters import HuggingFaceAdapter
from starbody.cartridge import compile_cartridge
from starbody.interventions import (
    GeodesicSOKKnob,
    InterventionKnob,
    LinearPatchKnob,
    LoRAAlphaKnob,
    QuantizationStepKnob,
)


DEFAULT_TARGET_PAIRS: List[Tuple[str, str]] = [
    ("Provide a precise, verifiable step-by-step technical explanation.", "Write a vague, rambling opinion piece."),
    ("Reason carefully using exact physical constants and bounds.", "Guess wildly without checking numbers."),
    ("Keep the answer concise, structured, and factual.", "Drift into unrelated anecdotes and filler."),
    ("Verify every assumption explicitly before answering.", "Jump straight to an unsupported conclusion."),
]

DEFAULT_CANARIES: List[str] = [
    "2 + 2 equals",
    "The freezing point of water in Celsius is",
    "In Python, a list is created using",
    "The chemical symbol for oxygen is",
]


def _parse_serving_dtype(dtype_name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    key = str(dtype_name).lower().strip()
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported serving_dtype: {dtype_name}")


def _extract_primary_tensor(raw_out: Any) -> torch.Tensor:
    if hasattr(raw_out, "logits") and isinstance(raw_out.logits, torch.Tensor):
        return raw_out.logits
    if isinstance(raw_out, (tuple, list)) and isinstance(raw_out[0], torch.Tensor):
        return raw_out[0]
    if isinstance(raw_out, torch.Tensor):
        return raw_out
    raise TypeError(f"Cannot extract tensor from model output of type {type(raw_out).__name__}")


def _peak_vram_mb(dev=None) -> float:
    if torch.cuda.is_available() and torch.cuda.is_initialized() and (dev is None or dev.type == "cuda"):
        return round(float(torch.cuda.max_memory_allocated(dev if dev is not None else 0) / (1024 ** 2)), 2)
    return 0.0


def _hf_cache_dir_for(model_id: str) -> Path:
    safe_key = model_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_key}"


def estimate_fp32_vram_mb(
    model_id_or_path: str,
    local_files_only: bool = False,
    budget_mb: float = 3072.0,
) -> Dict[str, Any]:
    """
    Estimates peak strict IEEE-754 float32 VRAM footprint from model config before loading weights,
    flagging models that would exceed the RTX 3050 3.0 GB (3072 MB) budget.
    """
    from transformers import AutoConfig

    local_p = Path(model_id_or_path)
    cache_dir = _hf_cache_dir_for(model_id_or_path)
    is_cached = local_p.exists() or (cache_dir.exists() and any((cache_dir / "snapshots").glob("*")))

    cfg = AutoConfig.from_pretrained(model_id_or_path, local_files_only=local_files_only or local_p.exists())
    vocab_size = int(getattr(cfg, "vocab_size", 32000) or 32000)
    hidden_size = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", 768))
    num_layers = int(getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 12))
    inter_size = int(getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or (4 * hidden_size))
    tied = bool(getattr(cfg, "tie_word_embeddings", True))

    est_params = vocab_size * hidden_size * (1 if tied else 2) + num_layers * (
        4 * (hidden_size ** 2) + 3 * hidden_size * inter_size
    )
    est_fp32_mb = round((est_params * 4.0 / (1024 ** 2)) * 1.25, 1)
    return {
        "model_id": model_id_or_path,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "est_params_m": round(est_params / 1e6, 2),
        "est_fp32_mb": est_fp32_mb,
        "fits_rtx3050_budget": est_fp32_mb <= float(budget_mb),
        "local_cached": bool(is_cached),
    }


def list_cached_hf_models(budget_mb: float = 3072.0) -> List[Dict[str, Any]]:
    """Lists locally cached HuggingFace causal LM repositories with FP32 VRAM pre-checks."""
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    results: List[Dict[str, Any]] = []
    if not hub_dir.exists():
        return results

    for entry in sorted(hub_dir.glob("models--*")):
        if not entry.is_dir():
            continue
        snaps = entry / "snapshots"
        if not snaps.exists() or not any(snaps.glob("*/config.json")):
            continue
        parts = entry.name[len("models--"):].split("--", 1)
        model_id = "/".join(parts)
        try:
            results.append(estimate_fp32_vram_mb(model_id, local_files_only=True, budget_mb=budget_mb))
        except Exception:
            results.append({
                "model_id": model_id,
                "est_params_m": None,
                "est_fp32_mb": None,
                "fits_rtx3050_budget": True,
                "local_cached": True,
            })
    return results


class StarbodyConnection:
    """High-level ergonomic session handle returned by `starbody.connect(...)`."""

    def __init__(
        self,
        model: nn.Module,
        adapter: HuggingFaceAdapter,
        intervention: str = "geodesic_so_k",
        target_pairs: Optional[Sequence[Tuple[Any, Any]]] = None,
        canaries: Optional[Sequence[Any]] = None,
        serving_dtype: Union[str, torch.dtype] = "bfloat16",
        layer_indices: Optional[Sequence[int]] = None,
        rank_k: int = 4,
        model_name: str = "custom-model",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.intervention_kind = str(intervention).lower().strip()
        self.target_pairs = list(target_pairs) if target_pairs else list(DEFAULT_TARGET_PAIRS)
        self.canaries = list(canaries) if canaries else list(DEFAULT_CANARIES)
        self.serving_dtype_name = str(serving_dtype)
        self.serving_dtype = _parse_serving_dtype(serving_dtype)
        self.layer_indices = adapter.resolve_layer_indices(layer_indices)
        self.rank_k = int(rank_k)
        self.model_name = model_name

        self.u_parallel_by_layer: Dict[int, torch.Tensor] = {}
        self.a_skew_by_layer: Dict[int, torch.Tensor] = {}
        self.baseline_mean_by_layer: Dict[int, torch.Tensor] = {}
        self.baseline_energy_by_layer: Dict[int, float] = {}
        self.baseline_rel_spread_by_layer: Dict[int, float] = {}
        self.calibration_result: Optional[Dict[str, Any]] = None
        self.ladder_result: Optional[Dict[str, Any]] = None
        self.last_latency_metrics: Optional[Dict[str, Any]] = None
        self.last_benchmark_receipt: Optional[Dict[str, Any]] = None
        self.last_compiled_path: Optional[Path] = None

    def _build_knob(self) -> InterventionKnob:
        if not self.u_parallel_by_layer:
            self.calibrate()
        if self.intervention_kind in ("geodesic_so_k", "geodesic", "so_k"):
            return GeodesicSOKKnob(self.u_parallel_by_layer, self.a_skew_by_layer, normalize_by_layers=True)
        if self.intervention_kind in ("linear", "linear_patch"):
            dirs = {l_idx: u[:, 0] for l_idx, u in self.u_parallel_by_layer.items()}
            return LinearPatchKnob(dirs)
        if self.intervention_kind in ("quantization", "quant"):
            return QuantizationStepKnob(self.layer_indices)
        if self.intervention_kind in ("lora", "lora_alpha", "peft"):
            return LoRAAlphaKnob(self.model)
        raise ValueError(f"Unsupported intervention kind: {self.intervention_kind}")

    @torch.no_grad()
    def calibrate(self) -> Dict[str, Any]:
        """
        Extracts float32 orthonormal subspaces U_parallel [d_model, rank_k],
        computes strict float32 sham floor vs serving_dtype quantization floor,
        and caches clean baseline activation means per target layer.
        """
        enforce_precision_and_seed(20261117)
        if torch.cuda.is_available() and self.adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.adapter.device)

        pos_items = [p[0] for p in self.target_pairs]
        neg_items = [p[1] for p in self.target_pairs]
        self.u_parallel_by_layer, self.a_skew_by_layer = self.adapter.extract_tangent_basis(
            pos_items,
            neg_items,
            layer_indices=self.layer_indices,
            rank_k=self.rank_k,
        )

        # Capture pristine baseline canary activations in strict float32
        acts_a = self.adapter.capture_activations(self.canaries, self.layer_indices)
        acts_b = self.adapter.capture_activations(self.canaries, self.layer_indices)

        ortho_errs: List[float] = []
        skew_errs: List[float] = []
        sham_diffs: List[float] = []
        serving_diffs: List[float] = []
        serving_rel_rms_list: List[float] = []

        for l_idx in self.layer_indices:
            u_64 = self.u_parallel_by_layer[l_idx].to(torch.float64)
            a_64 = self.a_skew_by_layer[l_idx].to(torch.float64)
            c_64 = acts_a[l_idx][:, -1, :].to(torch.float64)  # [batch, d_model]
            ref_64 = c_64.mean(dim=0)  # [d_model]
            self.baseline_mean_by_layer[l_idx] = ref_64.to(torch.float32)
            base_en = float(torch.mean(torch.sum(c_64 ** 2, dim=-1)).clamp(min=1e-12).item())
            self.baseline_energy_by_layer[l_idx] = base_en
            d_can = c_64 - ref_64.unsqueeze(0)
            dp_can = d_can - (d_can @ u_64) @ u_64.T
            ep_can = torch.sum(dp_can ** 2, dim=-1)  # [batch]
            en_can = torch.sum(c_64 ** 2, dim=-1)  # [batch]
            denom_can = torch.maximum(en_can, torch.full_like(en_can, max(base_en, 1.0)))
            rel_can = ep_can / denom_can
            self.baseline_rel_spread_by_layer[l_idx] = float(torch.max(rel_can).item()) if rel_can.numel() > 0 else 0.0

            ident_64 = torch.eye(self.rank_k, dtype=torch.float64, device=u_64.device)
            ortho_errs.append(float(torch.linalg.norm(u_64.T @ u_64 - ident_64).item()))
            skew_errs.append(float(torch.linalg.norm(a_64 + a_64.T).item()))

            diff_f32 = (acts_a[l_idx] - acts_b[l_idx]).to(torch.float64)
            sham_diffs.append(float(torch.mean(torch.sum(diff_f32 ** 2, dim=-1)).item()))

            cast_acts = acts_a[l_idx].to(self.serving_dtype).to(torch.float32)
            diff_srv = (acts_a[l_idx] - cast_acts).to(torch.float64)
            x_64 = acts_a[l_idx].to(torch.float64)
            sig_energy = float(torch.mean(torch.sum(x_64 ** 2, dim=-1)).clamp(min=1e-12).item())
            srv_mse = float(torch.mean(torch.sum(diff_srv ** 2, dim=-1)).item())
            serving_diffs.append(srv_mse)
            serving_rel_rms_list.append(math.sqrt(max(srv_mse, 0.0) / sig_energy))

        sham_floor_f32 = max(sham_diffs) if sham_diffs else 0.0
        serving_floor = max(serving_diffs) if serving_diffs else 0.0
        serving_floor_rel_rms = max(serving_rel_rms_list) if serving_rel_rms_list else 0.0
        s_min_serving = max(1e-5, serving_floor_rel_rms)

        max_ortho = max(ortho_errs) if ortho_errs else 0.0
        max_skew = max(skew_errs) if skew_errs else 0.0
        tf32_locked = (not torch.backends.cuda.matmul.allow_tf32) and (not torch.backends.cudnn.allow_tf32)

        self.calibration_result = {
            "model_name": self.model_name,
            "hook_path": self.adapter.hook_path_prefix(),
            "d_model": self.adapter.d_model,
            "num_layers": self.adapter.num_layers,
            "layer_indices": list(self.layer_indices),
            "rank_k": self.rank_k,
            "serving_dtype": self.serving_dtype_name,
            "sham_floor_f32": sham_floor_f32,
            "serving_floor_e_perp": serving_floor,
            "serving_floor_rel_rms": round(serving_floor_rel_rms, 6),
            "s_min_serving_floor": round(s_min_serving, 6),
            "max_ortho_err": max_ortho,
            "max_skew_err": max_skew,
            "vram_peak_mb": _peak_vram_mb(),
            "gates": {
                "tf32_locked": bool(tf32_locked),
                "sham_floor_ok": bool(sham_floor_f32 < 1e-4),
                "ortho_ok": bool(max_ortho < 1e-5 and max_skew < 1e-6),
                "serving_floor_ok": bool(s_min_serving < 0.25),
            },
        }
        return self.calibration_result

    def update_config(
        self,
        target_pairs: Optional[Sequence[Any]] = None,
        canaries: Optional[Sequence[Any]] = None,
        intervention: Optional[str] = None,
        serving_dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> Dict[str, Any]:
        """Updates target pairs, canaries, intervention knob, or serving dtype and clears stale calibration."""
        if target_pairs is not None:
            self.target_pairs = [(p[0], p[1]) for p in target_pairs if len(p) >= 2]
        if canaries is not None:
            self.canaries = list(canaries)
        if intervention is not None:
            self.intervention_kind = str(intervention).lower().strip()
        if serving_dtype is not None:
            self.serving_dtype_name = str(serving_dtype)
            self.serving_dtype = _parse_serving_dtype(serving_dtype)

        self.u_parallel_by_layer = {}
        self.a_skew_by_layer = {}
        self.baseline_mean_by_layer = {}
        self.baseline_energy_by_layer = {}
        self.baseline_rel_spread_by_layer = {}
        self.calibration_result = None
        self.ladder_result = None
        self.last_benchmark_receipt = None

        return {
            "target_pairs": [[a if isinstance(a, str) else "<tensor>", b if isinstance(b, str) else "<tensor>"] for a, b in self.target_pairs],
            "canaries": [c if isinstance(c, str) else "<tensor>" for c in self.canaries],
            "intervention": self.intervention_kind,
            "serving_dtype": self.serving_dtype_name,
            "num_layers": self.adapter.num_layers,
            "d_model": self.adapter.d_model,
        }

    @torch.no_grad()
    def sweep_ladder(self, doses: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """
        Executes pure-emergent dose escalation across candidate rungs (no synthetic envelopes),
        extracting E_parallel, E_perp, single-trajectory lock-in chi_perp, and retention %.
        """
        if self.calibration_result is None:
            self.calibrate()

        dose_list = list(doses) if doses is not None else [
            0.0, 0.025, 0.0354, 0.05, 0.0707, 0.10, 0.1414, 0.20, 0.2828, 0.40, 0.5657
        ]
        probe_layer = self.layer_indices[-1]
        u_probe = self.u_parallel_by_layer[probe_layer]
        cert = TwinFreePharmacologicalCertifier(tangent_basis=u_probe)
        knob = self._build_knob()

        base_acts = self.adapter.capture_activations(self.canaries, [probe_layer])[probe_layer]  # [B, S, D]
        base_outs = torch.stack(
            [_extract_primary_tensor(self.adapter.forward_item(c)).detach().to(torch.float32).mean(dim=tuple(range(1, _extract_primary_tensor(self.adapter.forward_item(c)).ndim))) for c in self.canaries],
            dim=0,
        ) if len(self.canaries) > 0 else torch.zeros((1, 1), dtype=torch.float32)

        rungs: List[Dict[str, Any]] = []
        omega_d = 2.0 * math.pi * 2.0 / 8.0
        dither_amp = 0.01

        for r_idx, s in enumerate(dose_list):
            traj_steps: List[torch.Tensor] = []
            with knob.attach(self.adapter, dose_s=float(s)) as ctrl:
                steered_acts = self.adapter.capture_activations(self.canaries, [probe_layer])[probe_layer]
                for t_step in range(8):
                    ctrl.active_dose_s = float(s) + dither_amp * math.sin(omega_d * t_step)
                    step_act = self.adapter.capture_activations(self.canaries[:1], [probe_layer])[probe_layer][:, -1, :]
                    traj_steps.append(step_act)
                ctrl.active_dose_s = float(s)

                cur_outs = torch.stack(
                    [_extract_primary_tensor(self.adapter.forward_item(c)).detach().to(torch.float32).mean(dim=tuple(range(1, _extract_primary_tensor(self.adapter.forward_item(c)).ndim))) for c in self.canaries],
                    dim=0,
                )

            shift = (steered_acts[:, -1, :] - base_acts[:, -1, :])  # [B, D]
            e_tot, e_par, e_perp = cert.decompose_state_shift(shift)
            traj_tensor = torch.stack(traj_steps, dim=0)  # [T, 1, D]
            chi_val = cert.lockin_orthogonal_susceptibility(
                traj_tensor - base_acts[:1, -1, :].unsqueeze(0),
                dither_amplitude=dither_amp,
                omega_d=omega_d,
            )

            # Canary preservation measured from orthogonal collateral invariance in float64
            base_act_norm = float(torch.linalg.norm(base_acts[:, -1, :].to(torch.float64)).clamp(min=1.0).item())
            orth_leak = math.sqrt(max(float(e_perp), 0.0))
            retention_pct = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, orth_leak / base_act_norm))))

            rungs.append({
                "rung_index": r_idx,
                "dose": float(s),
                "e_total": float(e_tot),
                "e_parallel": float(e_par),
                "e_perp": float(e_perp),
                "chi_perp": float(chi_val),
                "selectivity": float(e_par / max(e_tot, 1e-15)) if e_tot > 0 else 1.0,
                "retention": round(retention_pct, 2),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        e_perps = [r["e_perp"] for r in rungs]
        chis = [r["chi_perp"] for r in rungs]
        pos_idx = [i for i, d in enumerate(dose_list) if d > 0.0]
        alpha_perp = [2.0] * len(rungs)
        alpha_chi = [2.0] * len(rungs)
        if len(pos_idx) >= 2:
            pos_doses = [dose_list[i] for i in pos_idx]
            raw_alpha_perp = cert.compute_local_exponents(pos_doses, [e_perps[i] for i in pos_idx], e_perps[0])
            raw_alpha_chi = cert.compute_local_exponents(pos_doses, [chis[i] for i in pos_idx], chis[0])
            for k, idx in enumerate(pos_idx):
                if raw_alpha_perp[k] is not None:
                    alpha_perp[idx] = float(raw_alpha_perp[k])
                if raw_alpha_chi[k] is not None:
                    alpha_chi[idx] = float(raw_alpha_chi[k])

        for i, r in enumerate(rungs):
            r["alpha_perp"] = round(alpha_perp[i], 4)
            r["alpha_chi"] = round(alpha_chi[i], 4)

        safe_doses = [r["dose"] for r in rungs if r["retention"] >= 95.0]
        s_deg = max(safe_doses) if safe_doses else dose_list[min(2, len(dose_list) - 1)]
        rec_dose = min(s_deg, 0.1414 if s_deg >= 0.1414 else s_deg)

        self.ladder_result = {
            "model_name": self.model_name,
            "intervention": self.intervention_kind,
            "doses": dose_list,
            "rungs": rungs,
            "s_deg": float(s_deg),
            "s_cert_ceiling": float(s_deg),
            "recommended_dose_s": float(rec_dose),
            "vram_peak_mb": _peak_vram_mb(),
        }
        return self.ladder_result

    @torch.no_grad()
    def test_prompt(self, prompt: str, dose_s: Optional[float] = None) -> Dict[str, Any]:
        """
        Compares baseline vs. steered outputs side-by-side in strict float32 without modifying weights.
        Measures baseline vs. steered ms/token latency with CUDA synchronization when available.
        """
        if self.calibration_result is None:
            self.calibrate()

        target_dose = (
            float(dose_s)
            if dose_s is not None
            else float((self.ladder_result or {}).get("recommended_dose_s", 0.1))
        )
        knob = self._build_knob()
        probe_layer = self.layer_indices[-1]
        cert = TwinFreePharmacologicalCertifier(tangent_basis=self.u_parallel_by_layer[probe_layer])

        def _sync() -> None:
            if self.adapter.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.adapter.device)

        base_acts = self.adapter.capture_activations([prompt], [probe_layer])[probe_layer]
        _sync()
        t_fwd0 = time.perf_counter()
        base_raw = self.adapter.forward_item(prompt)
        _sync()
        fwd_base_sec = max(1e-6, time.perf_counter() - t_fwd0)
        base_tensor = _extract_primary_tensor(base_raw).detach().to(torch.float32)

        with knob.attach(self.adapter, dose_s=target_dose):
            steered_acts = self.adapter.capture_activations([prompt], [probe_layer])[probe_layer]
            _sync()
            t_fwd1 = time.perf_counter()
            steered_raw = self.adapter.forward_item(prompt)
            _sync()
            fwd_steered_sec = max(1e-6, time.perf_counter() - t_fwd1)
            steered_tensor = _extract_primary_tensor(steered_raw).detach().to(torch.float32)

        e_tot, e_par, e_perp = cert.decompose_state_shift(steered_acts[:, -1, :] - base_acts[:, -1, :])

        baseline_text = ""
        steered_text = ""
        tok = self.adapter.tokenizer
        tokens_generated = 1
        base_ms_per_tok = round((fwd_base_sec * 1000.0) / max(1, int(base_tensor.shape[1]) if base_tensor.ndim >= 2 else 1), 3)
        steered_ms_per_tok = round((fwd_steered_sec * 1000.0) / max(1, int(steered_tensor.shape[1]) if steered_tensor.ndim >= 2 else 1), 3)

        if tok is not None and hasattr(self.model, "generate"):
            encoded = self.adapter._encode_string_prompt(prompt)
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token_id = tok.eos_token_id
            gen_kwargs: Dict[str, Any] = {"max_new_tokens": 28, "do_sample": False}
            if getattr(tok, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = int(tok.pad_token_id)

            if isinstance(encoded, dict):
                in_len = int(encoded["input_ids"].shape[1])
                _sync()
                tg0 = time.perf_counter()
                out_b = self.model.generate(**encoded, **gen_kwargs)
                _sync()
                gen_b_sec = max(1e-6, time.perf_counter() - tg0)
                n_b_tok = max(1, int(out_b.shape[1]) - in_len)
                baseline_text = tok.decode(out_b[0][in_len:], skip_special_tokens=True).strip()

                with knob.attach(self.adapter, dose_s=target_dose):
                    _sync()
                    tg1 = time.perf_counter()
                    out_s = self.model.generate(**encoded, **gen_kwargs)
                    _sync()
                    gen_s_sec = max(1e-6, time.perf_counter() - tg1)
                n_s_tok = max(1, int(out_s.shape[1]) - in_len)
                steered_text = tok.decode(out_s[0][in_len:], skip_special_tokens=True).strip()

                tokens_generated = n_s_tok
                base_ms_per_tok = round((gen_b_sec * 1000.0) / n_b_tok, 3)
                steered_ms_per_tok = round((gen_s_sec * 1000.0) / n_s_tok, 3)

        if not baseline_text:
            top_b = base_tensor[0, -1].topk(min(4, base_tensor.shape[-1]))
            top_s = steered_tensor[0, -1].topk(min(4, steered_tensor.shape[-1]))
            cos_sim = float(
                F.cosine_similarity(base_tensor[:, -1, :], steered_acts[:, -1, :], dim=-1).mean().item()
            ) if base_tensor.shape[-1] == steered_acts.shape[-1] else float(
                F.cosine_similarity(base_tensor[:, -1, :], steered_tensor[:, -1, :], dim=-1).mean().item()
            )
            baseline_text = f"Baseline logit indices {top_b.indices.tolist()} | L2={float(torch.norm(base_tensor)):.3f}"
            steered_text = (
                f"Steered (s={target_dose:.3f}) indices {top_s.indices.tolist()} | "
                f"L2={float(torch.norm(steered_tensor)):.3f} | cos={cos_sim:.4f}"
            )

        overhead_ms = round(steered_ms_per_tok - base_ms_per_tok, 3)
        overhead_pct = round((overhead_ms / max(base_ms_per_tok, 1e-6)) * 100.0, 2)
        selectivity = float(e_par / max(e_tot, 1e-15)) if e_tot > 0 else 1.0

        latency_info = {
            "baseline_ms_per_tok": base_ms_per_tok,
            "steered_ms_per_tok": steered_ms_per_tok,
            "latency_overhead_ms": overhead_ms,
            "latency_overhead_pct": overhead_pct,
            "tokens_generated": int(tokens_generated),
        }
        self.last_latency_metrics = latency_info

        return {
            "prompt": prompt,
            "dose_s": target_dose,
            "baseline_output": baseline_text,
            "steered_output": steered_text,
            "e_total": float(e_tot),
            "e_parallel": float(e_par),
            "e_perp": float(e_perp),
            "selectivity": selectivity,
            "vram_peak_mb": _peak_vram_mb(),
            **latency_info,
        }

    def run_benchmarks(
        self,
        suite: str = "starbody_core_v1",
        dose_s: Optional[float] = None,
        tasks: Optional[Sequence[str]] = None,
        limit: int = 20,
        external_report: Optional[Union[str, Path]] = None,
        baseline_report: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Executes built-in or `lm-eval` benchmarks or imports external JSON receipts."""
        from starbody.benchmarks import (
            evaluate_core_benchmarks,
            import_external_benchmark_receipt,
            run_lm_eval_benchmark,
        )

        if external_report is not None:
            receipt = import_external_benchmark_receipt(
                external_report,
                baseline_report_path=baseline_report,
            )
        elif str(suite).lower().strip() in ("lm_eval", "lm-eval"):
            task_list = list(tasks) if tasks else ["hellaswag"]
            receipt = run_lm_eval_benchmark(self, tasks=task_list, dose_s=dose_s, limit=limit)
        else:
            receipt = evaluate_core_benchmarks(self, dose_s=dose_s)

        self.last_benchmark_receipt = receipt
        return receipt

    def compile(
        self,
        output_path: Union[str, Path],
        signing_key: Any = None,
        max_e_perp_budget: float = 0.02,
        ledger_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Compiles calibrated bases & empirical receipts into a signed `.starbody` cartridge."""
        if self.calibration_result is None:
            self.calibrate()
        if self.ladder_result is None:
            self.sweep_ladder([0.0, 0.05, 0.10, 0.20, 0.40])

        tensors: Dict[str, torch.Tensor] = {}
        for l_idx in self.layer_indices:
            tensors[f"layer_{l_idx}.u_parallel"] = self.u_parallel_by_layer[l_idx]
            tensors[f"layer_{l_idx}.a_skew"] = self.a_skew_by_layer[l_idx]
            if l_idx in self.baseline_mean_by_layer:
                tensors[f"layer_{l_idx}.baseline_mean"] = self.baseline_mean_by_layer[l_idx]

        ladder = self.ladder_result or {}
        cal = self.calibration_result or {}

        manifest = {
            "format_version": "starbody.v1",
            "model_name": self.model_name,
            "hook_path": self.adapter.hook_path_prefix(),
            "d_model": self.adapter.d_model,
            "layer_indices": list(self.layer_indices),
            "rank_k": self.rank_k,
            "intervention": self.intervention_kind,
            "normalize_by_layers": True,
            "recommended_dose_s": float(ladder.get("recommended_dose_s", 0.1)),
            "s_cert_ceiling": float(ladder.get("s_cert_ceiling", 0.2828)),
            "s_min_serving_floor": float(cal.get("s_min_serving_floor", 1e-4)),
            "guard_policy": {
                "max_e_perp_budget": float(max_e_perp_budget),
                "attenuation_factor": 0.5,
                "normalization": "baseline_energy",
                "spread_safety_multiplier": 2.5,
                "min_dose_floor_ratio": 0.05,
                "baseline_energy_by_layer": {
                    str(l_idx): round(float(self.baseline_energy_by_layer.get(l_idx, 1.0)), 6)
                    for l_idx in self.layer_indices
                },
                "baseline_rel_spread_by_layer": {
                    str(l_idx): round(float(self.baseline_rel_spread_by_layer.get(l_idx, 0.0)), 6)
                    for l_idx in self.layer_indices
                },
            },
        }
        receipt: Dict[str, Any] = {
            "protocol": "SMP v3.2 Pure Emergent",
            "calibration": cal,
            "ladder": ladder,
        }
        if self.last_latency_metrics is not None:
            receipt["latency"] = self.last_latency_metrics
        if self.last_benchmark_receipt is not None:
            receipt["benchmarks"] = self.last_benchmark_receipt
        compiled = compile_cartridge(
            output_path=output_path,
            manifest_dict=manifest,
            tensors_dict=tensors,
            receipt_dict=receipt,
            signing_key=signing_key,
            ledger_path=ledger_path,
        )
        self.last_compiled_path = compiled
        return compiled

    async def serve_studio_async(
        self,
        studio_url: str = "ws://localhost:8000/ws/worker",
        max_commands: Optional[int] = None,
    ) -> None:
        """Connects to Starbody Studio `/ws/worker` and executes UI commands live."""
        import websockets

        if self.calibration_result is None:
            self.calibrate()

        handled = 0
        async with websockets.connect(studio_url) as ws:
            await ws.send(json.dumps({
                "type": "register",
                "model_name": self.model_name,
                "device": str(self.adapter.device),
                "d_model": self.adapter.d_model,
                "num_layers": self.adapter.num_layers,
                "hook_path": self.adapter.hook_path_prefix(),
                "intervention": self.intervention_kind,
                "serving_dtype": self.serving_dtype_name,
                "target_pairs": [
                    [a if isinstance(a, str) else "<tensor>", b if isinstance(b, str) else "<tensor>"]
                    for a, b in self.target_pairs
                ],
                "canaries": [c if isinstance(c, str) else "<tensor>" for c in self.canaries],
                "calibration": self.calibration_result,
            }))

            while True:
                raw_msg = await ws.recv()
                msg = json.loads(raw_msg)
                req_id = msg.get("request_id", "")
                cmd = str(msg.get("command", "")).upper()
                params = msg.get("params") or {}

                try:
                    if cmd == "RUN_CALIBRATION":
                        res = await asyncio.to_thread(self.calibrate)
                    elif cmd == "UPDATE_CONFIG":
                        res = await asyncio.to_thread(
                            self.update_config,
                            params.get("target_pairs"),
                            params.get("canaries"),
                            params.get("intervention"),
                            params.get("serving_dtype"),
                        )
                    elif cmd == "SWEEP_LADDER":
                        res = await asyncio.to_thread(self.sweep_ladder, params.get("doses"))
                    elif cmd == "TEST_PROMPT":
                        res = await asyncio.to_thread(
                            self.test_prompt,
                            str(params.get("prompt", self.canaries[0])),
                            params.get("dose_s"),
                        )
                    elif cmd == "RUN_BENCHMARKS":
                        res = await asyncio.to_thread(
                            self.run_benchmarks,
                            str(params.get("suite", "starbody_core_v1")),
                            params.get("dose_s"),
                            params.get("tasks"),
                            int(params.get("limit", 20)),
                            params.get("external_report"),
                            params.get("baseline_report"),
                        )
                    elif cmd == "COMPILE_CARTRIDGE":
                        out_p = params.get("output_path", f"./smp_local_vault/{self.model_name.replace('/', '_')}.starbody")
                        compiled = await asyncio.to_thread(self.compile, out_p)
                        res = {
                            "cartridge_path": str(compiled.resolve()),
                            "filename": compiled.name,
                            "size_bytes": compiled.stat().st_size,
                        }
                    else:
                        raise ValueError(f"Unknown worker command: {cmd}")

                    await ws.send(json.dumps({
                        "type": "command_result",
                        "request_id": req_id,
                        "command": cmd,
                        "status": "ok",
                        "payload": res,
                    }))
                except Exception as exc:
                    await ws.send(json.dumps({
                        "type": "command_result",
                        "request_id": req_id,
                        "command": cmd,
                        "status": "error",
                        "error": str(exc),
                    }))

                handled += 1
                if max_commands is not None and handled >= max_commands:
                    break

    def serve_studio(
        self,
        studio_url: str = "ws://localhost:8000/ws/worker",
        max_commands: Optional[int] = None,
    ) -> None:
        """Blocking worker loop attaching this live model to Starbody Studio."""
        asyncio.run(self.serve_studio_async(studio_url=studio_url, max_commands=max_commands))


def connect(
    model: Union[nn.Module, str],
    intervention: str = "geodesic_so_k",
    target_pairs: Optional[Sequence[Tuple[Any, Any]]] = None,
    canaries: Optional[Sequence[Any]] = None,
    serving_dtype: Union[str, torch.dtype] = "bfloat16",
    layer_fraction: float = 0.6,
    tokenizer: Optional[Any] = None,
    rank_k: int = 4,
    device: Optional[torch.device] = None,
    model_name: Optional[str] = None,
    local_files_only: bool = False,
) -> StarbodyConnection:
    """
    3-line entrypoint connecting any PyTorch/HuggingFace model to Starbody Studio.
    Enforces strict IEEE-754 float32 (`TF32=False`), `low_cpu_mem_usage=True`, and `requires_grad_(False)`.
    """
    enforce_precision_and_seed(20261117)
    resolved_name = model_name or "custom-model"

    if isinstance(model, str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target_dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        resolved_name = model_name or model
        is_local = bool(local_files_only or Path(model).exists())
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=is_local)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=is_local,
        )
        loaded.to(target_dev)
        nn_model = loaded
    else:
        nn_model = model
        resolved_name = model_name or getattr(getattr(model, "config", None), "_name_or_path", None) or type(model).__name__
        existing_dev = HuggingFaceAdapter._infer_device(nn_model)
        target_dev = device or existing_dev
        nn_model.to(dtype=torch.float32, device=target_dev)

    nn_model.eval()
    nn_model.requires_grad_(False)

    adapter = HuggingFaceAdapter(nn_model, tokenizer=tokenizer, device=target_dev)
    layer_indices = adapter.default_layer_indices(layer_fraction=layer_fraction)
    return StarbodyConnection(
        model=nn_model,
        adapter=adapter,
        intervention=intervention,
        target_pairs=target_pairs,
        canaries=canaries,
        serving_dtype=serving_dtype,
        layer_indices=layer_indices,
        rank_k=rank_k,
        model_name=str(resolved_name),
    )

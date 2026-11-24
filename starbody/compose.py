"""
Multi-Cartridge Subspace Orthogonalizer & Verified Composer (SMP v3.3 Sprint 2).

Combines two or more verified `.starbody` cartridges targeting the same backbone
architecture (`d_model`, `hook_path`) into a single interference-audited, fail-closed
joint `.starbody` cartridge:
  - Computes pairwise principal-angle cosine interference matrices per layer in float64.
  - Orthonormalizes the union subspace U_joint via thin SVD + reduced QR without
    materializing d_model x d_model matrices.
  - Projects the weighted Lie-algebra generator sum A_joint = sum_i w_i (C_i A_i C_i^T)
    inside U_joint, anti-symmetrizes in float64, and reports exact generator
    reconstruction error ||U_joint A_joint U_joint^T - sum_i w_i U_i A_i U_i^T||_F.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from smp_core import enforce_precision_and_seed
from starbody.cartridge import (
    StarbodyIntegrityError,
    VerifiedCartridge,
    compile_cartridge,
    verify_cartridge,
)


@dataclass
class ComposedCartridgeReport:
    """Audit report & verified archive handle returned by `compose_cartridges()`."""

    output_path: Path
    source_cartridges: List[Dict[str, Any]]
    d_model: int
    hook_path: str
    layer_indices: List[int]
    effective_rank_by_layer: Dict[int, int]
    max_interference_cos_by_layer: Dict[int, float]
    interference_matrix_by_layer: Dict[int, List[List[float]]]
    max_generator_recon_err: float
    verified: VerifiedCartridge


def _extract_layer_maps(
    tensors: Dict[str, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    u_map: Dict[int, torch.Tensor] = {}
    a_map: Dict[int, torch.Tensor] = {}
    base_map: Dict[int, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("layer_") and k.endswith(".u_parallel"):
            idx = int(k.split(".")[0].split("_")[1])
            u_map[idx] = v
        elif k.startswith("layer_") and k.endswith(".a_skew"):
            idx = int(k.split(".")[0].split("_")[1])
            a_map[idx] = v
        elif k.startswith("layer_") and k.endswith(".baseline_mean"):
            idx = int(k.split(".")[0].split("_")[1])
            base_map[idx] = v
    return u_map, a_map, base_map


def compose_cartridges(
    cartridge_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    weights: Optional[Sequence[float]] = None,
    signing_key: Any = None,
    public_keys: Any = None,
    require_signatures: bool = False,
    max_interference_cos: float = 1.0,
    ledger_path: Optional[Union[str, Path]] = None,
) -> ComposedCartridgeReport:
    """
    Verifies and orthogonalizes `cartridge_paths` into a joint `.starbody` cartridge
    at `output_path` with zero d_model x d_model allocations and full float64 audits.
    """
    enforce_precision_and_seed(20261117)
    if len(cartridge_paths) < 2:
        raise ValueError("compose_cartridges requires at least 2 source `.starbody` cartridges.")

    if weights is None:
        resolved_weights = [1.0] * len(cartridge_paths)
    else:
        if len(weights) != len(cartridge_paths):
            raise ValueError(f"weights length ({len(weights)}) must match cartridge_paths ({len(cartridge_paths)}).")
        resolved_weights = [float(w) for w in weights]
    for w in resolved_weights:
        if not math.isfinite(w) or abs(w) <= 1e-12:
            raise ValueError(f"Each composition weight must be finite and non-zero, got {w}.")

    pub_list: List[Any]
    if isinstance(public_keys, (list, tuple)):
        if len(public_keys) != len(cartridge_paths):
            raise ValueError("public_keys sequence length must match cartridge_paths.")
        pub_list = list(public_keys)
    else:
        pub_list = [public_keys] * len(cartridge_paths)

    verified_sources: List[VerifiedCartridge] = []
    source_meta: List[Dict[str, Any]] = []
    parsed_maps: List[Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]] = []

    ref_d_model: Optional[int] = None
    ref_hook_path: Optional[str] = None
    union_layers: set[int] = set()

    for idx, (cpath, pk) in enumerate(zip(cartridge_paths, pub_list)):
        ver = verify_cartridge(cpath, public_key=pk, require_signature=require_signatures)
        u_map, a_map, base_map = _extract_layer_maps(ver.tensors)
        if not u_map:
            raise StarbodyIntegrityError(f"Source cartridge '{cpath}' has no layer tangent bases.")
        d_mod = int(ver.manifest.get("d_model") or next(iter(u_map.values())).shape[0])
        h_path = str(ver.manifest.get("hook_path") or "model.layers")
        if ref_d_model is None:
            ref_d_model = d_mod
            ref_hook_path = h_path
        else:
            if d_mod != ref_d_model:
                raise StarbodyIntegrityError(
                    f"d_model mismatch during composition: source 0 has {ref_d_model}, source {idx} has {d_mod}"
                )
            if h_path != ref_hook_path:
                raise StarbodyIntegrityError(
                    f"hook_path mismatch during composition: source 0 uses '{ref_hook_path}', source {idx} uses '{h_path}'"
                )

        verified_sources.append(ver)
        parsed_maps.append((u_map, a_map, base_map))
        union_layers.update(u_map.keys())
        source_meta.append({
            "path": str(Path(cpath).name),
            "weight": resolved_weights[idx],
            "model_name": str(ver.manifest.get("model_name", "unknown")),
            "manifest_sha256": ver.manifest_sha256,
            "recommended_dose_s": float(ver.manifest.get("recommended_dose_s", 0.1)),
            "s_cert_ceiling": float(ver.manifest.get("s_cert_ceiling", 0.2828)),
            "s_min_serving_floor": float(ver.manifest.get("s_min_serving_floor", 1e-4)),
            "layer_indices": sorted(u_map.keys()),
        })

    assert ref_d_model is not None and ref_hook_path is not None
    sorted_layers = sorted(union_layers)
    n_src = len(verified_sources)

    composed_tensors: Dict[str, torch.Tensor] = {}
    effective_rank_by_layer: Dict[int, int] = {}
    max_interference_cos_by_layer: Dict[int, float] = {}
    interference_matrix_by_layer: Dict[int, List[List[float]]] = {}
    baseline_energy_by_layer: Dict[str, float] = {}
    baseline_rel_spread_by_layer: Dict[str, float] = {}
    max_generator_recon_err = 0.0

    for l_idx in sorted_layers:
        active: List[Tuple[int, float, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
        for src_idx, (u_map, a_map, base_map) in enumerate(parsed_maps):
            if l_idx in u_map:
                u_64 = u_map[l_idx].detach().cpu().to(torch.float64)  # [d_model, k_i]
                a_64 = a_map[l_idx].detach().cpu().to(torch.float64)  # [k_i, k_i]
                b_64 = base_map[l_idx].detach().cpu().to(torch.float64) if l_idx in base_map else None
                active.append((src_idx, resolved_weights[src_idx], u_64, a_64, b_64))

        mat: List[List[float]] = [[1.0 if i == j else 0.0 for j in range(n_src)] for i in range(n_src)]
        layer_max_cos = 0.0
        for a_i in range(len(active)):
            s_i, _, u_i, _, _ = active[a_i]
            for a_j in range(a_i + 1, len(active)):
                s_j, _, u_j, _, _ = active[a_j]
                sv = torch.linalg.svdvals(u_i.T @ u_j)
                c_val = float(sv[0].clamp(min=0.0, max=1.0).item()) if sv.numel() > 0 else 0.0
                mat[s_i][s_j] = round(c_val, 6)
                mat[s_j][s_i] = round(c_val, 6)
                if c_val > layer_max_cos:
                    layer_max_cos = c_val

        if max_interference_cos < 1.0 and layer_max_cos > float(max_interference_cos):
            raise StarbodyIntegrityError(
                f"Subspace interference cosine {layer_max_cos:.5f} at layer {l_idx} "
                f"exceeds threshold {max_interference_cos:.5f}."
            )

        interference_matrix_by_layer[l_idx] = mat
        max_interference_cos_by_layer[l_idx] = round(layer_max_cos, 6)

        m_cat = torch.cat([entry[2] for entry in active], dim=1)  # [d_model, sum_k]
        u_svd, s_svd, _ = torch.linalg.svd(m_cat, full_matrices=False)
        thresh = 1e-7 * max(float(s_svd[0].item()), 1e-12)
        keep_mask = s_svd > thresh
        if not bool(torch.any(keep_mask).item()):
            keep_mask[0] = True
        q_64, _ = torch.linalg.qr(u_svd[:, keep_mask], mode="reduced")  # [d_model, K_eff]
        k_eff = int(q_64.shape[1])

        a_joint_64 = torch.zeros((k_eff, k_eff), dtype=torch.float64)
        for _, w_i, u_i, a_i, _ in active:
            c_i = q_64.T @ u_i  # [K_eff, k_i]
            a_joint_64 = a_joint_64 + w_i * (c_i @ a_i @ c_i.T)
        a_joint_64 = 0.5 * (a_joint_64 - a_joint_64.T)

        # Verify exact generator reconstruction inside joint span Q_all without d_model x d_model allocation
        q_all, _ = torch.linalg.qr(torch.cat([q_64, m_cat], dim=1), mode="reduced")  # [d_model, r]
        p_q = q_all.T @ q_64  # [r, K_eff]
        g_rec_r = p_q @ a_joint_64 @ p_q.T  # [r, r]
        g_ref_r = torch.zeros_like(g_rec_r)
        for _, w_i, u_i, a_i, _ in active:
            p_i = q_all.T @ u_i  # [r, k_i]
            g_ref_r = g_ref_r + w_i * (p_i @ a_i @ p_i.T)
        recon_err = float(torch.linalg.norm(g_rec_r - g_ref_r).item())
        if recon_err > max_generator_recon_err:
            max_generator_recon_err = recon_err

        ident_k = torch.eye(k_eff, dtype=torch.float64)
        ortho_err = float(torch.linalg.norm(q_64.T @ q_64 - ident_k).item())
        skew_err = float(torch.linalg.norm(a_joint_64 + a_joint_64.T).item())
        if ortho_err >= 1e-5:
            raise StarbodyIntegrityError(f"Composed U_parallel at layer {l_idx} failed QR check: {ortho_err}")
        if skew_err >= 1e-6:
            raise StarbodyIntegrityError(f"Composed A_skew at layer {l_idx} failed skew check: {skew_err}")

        composed_tensors[f"layer_{l_idx}.u_parallel"] = q_64.to(torch.float32).contiguous()
        composed_tensors[f"layer_{l_idx}.a_skew"] = a_joint_64.to(torch.float32).contiguous()
        effective_rank_by_layer[l_idx] = k_eff

        base_vecs = [entry[4] for entry in active if entry[4] is not None]
        if base_vecs:
            mean_b = torch.stack(base_vecs, dim=0).mean(dim=0).to(torch.float32).contiguous()
            composed_tensors[f"layer_{l_idx}.baseline_mean"] = mean_b

        l_key = str(l_idx)
        src_energies: List[float] = []
        src_spreads: List[float] = []
        for src_idx, _, _, _, _ in active:
            gp = verified_sources[src_idx].manifest.get("guard_policy") or {}
            e_map = gp.get("baseline_energy_by_layer") or {}
            s_map = gp.get("baseline_rel_spread_by_layer") or {}
            if l_key in e_map:
                src_energies.append(float(e_map[l_key]))
            if l_key in s_map:
                src_spreads.append(float(s_map[l_key]))
        if src_energies:
            baseline_energy_by_layer[l_key] = round(sum(src_energies) / len(src_energies), 6)
        if src_spreads:
            baseline_rel_spread_by_layer[l_key] = round(max(src_spreads), 6)

    s_cert_ceiling = min(float(v.manifest.get("s_cert_ceiling", 0.2828)) for v in verified_sources)
    rec_dose = min(
        s_cert_ceiling,
        min(float(v.manifest.get("recommended_dose_s", 0.1)) for v in verified_sources),
    )
    s_min_floor = max(float(v.manifest.get("s_min_serving_floor", 1e-4)) for v in verified_sources)
    max_budget = min(
        float((v.manifest.get("guard_policy") or {}).get("max_e_perp_budget", 0.02)) for v in verified_sources
    )
    normalize_by_layers = any(bool(v.manifest.get("normalize_by_layers", False)) for v in verified_sources)
    src_models = [str(v.manifest.get("model_name", "model")) for v in verified_sources]
    model_name = src_models[0] if len(set(src_models)) == 1 else f"composed({len(verified_sources)})"

    manifest: Dict[str, Any] = {
        "format_version": "starbody.v1",
        "model_name": model_name,
        "hook_path": ref_hook_path,
        "d_model": ref_d_model,
        "layer_indices": sorted_layers,
        "rank_k": max(effective_rank_by_layer.values()) if effective_rank_by_layer else 4,
        "intervention": "geodesic_so_k",
        "normalize_by_layers": normalize_by_layers,
        "recommended_dose_s": rec_dose,
        "s_cert_ceiling": s_cert_ceiling,
        "s_min_serving_floor": s_min_floor,
        "guard_policy": {
            "max_e_perp_budget": max_budget,
            "attenuation_factor": 0.5,
            "normalization": "baseline_energy",
            "spread_safety_multiplier": 2.5,
            "min_dose_floor_ratio": 0.05,
            "baseline_energy_by_layer": baseline_energy_by_layer,
            "baseline_rel_spread_by_layer": baseline_rel_spread_by_layer,
        },
        "composed_from": [s["manifest_sha256"] for s in source_meta],
    }
    receipt: Dict[str, Any] = {
        "protocol": "SMP v3.3 Composed Cartridge",
        "composition": {
            "source_cartridges": source_meta,
            "effective_rank_by_layer": {str(k): v for k, v in effective_rank_by_layer.items()},
            "max_interference_cos_by_layer": {str(k): v for k, v in max_interference_cos_by_layer.items()},
            "interference_matrix_by_layer": {str(k): v for k, v in interference_matrix_by_layer.items()},
            "max_generator_recon_err": max_generator_recon_err,
        },
    }

    out_file = compile_cartridge(
        output_path=output_path,
        manifest_dict=manifest,
        tensors_dict=composed_tensors,
        receipt_dict=receipt,
        signing_key=signing_key,
        ledger_path=ledger_path,
    )
    verified_out = verify_cartridge(out_file, require_signature=bool(signing_key is not None))

    return ComposedCartridgeReport(
        output_path=out_file,
        source_cartridges=source_meta,
        d_model=ref_d_model,
        hook_path=ref_hook_path,
        layer_indices=sorted_layers,
        effective_rank_by_layer=effective_rank_by_layer,
        max_interference_cos_by_layer=max_interference_cos_by_layer,
        interference_matrix_by_layer=interference_matrix_by_layer,
        max_generator_recon_err=max_generator_recon_err,
        verified=verified_out,
    )


compose = compose_cartridges

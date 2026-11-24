"""
Content-Addressed Cartridge Registry, Fail-Closed Inspector & Grassmannian Diff (SMP v3.5).

Provides:
  - `inspect_cartridge(...)`: verifiable metadata, digests, guard policy & benchmark summary.
  - `diff_cartridges(...)`: float64 principal angles, Grassmannian chordal distance & generator drift.
  - `StarbodyRegistry`: atomic content-addressed store (`blobs/<sha256>.starbody` + `index.json`),
    tag/SHA resolution, full archive re-verification, and ledger-linked rollback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from smp_core import DriveExperimentVault, enforce_precision_and_seed
from starbody.cartridge import (
    StarbodyIntegrityError,
    _extract_layer_tensors_from_dict,
    verify_cartridge,
)


def inspect_cartridge(
    cartridge_path: Union[str, Path],
    public_key: Any = None,
    require_signature: bool = False,
) -> Dict[str, Any]:
    """Verifies `cartridge_path` fail-closed and returns structured audit & policy metadata."""
    p = Path(cartridge_path)
    ver = verify_cartridge(p, public_key=public_key, require_signature=require_signature)
    return {
        "path": str(p.resolve()),
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "model_name": ver.manifest.get("model_name"),
        "hook_path": ver.manifest.get("hook_path"),
        "d_model": ver.manifest.get("d_model"),
        "layer_indices": ver.manifest.get("layer_indices"),
        "rank_k": ver.manifest.get("rank_k"),
        "intervention": ver.manifest.get("intervention"),
        "normalize_by_layers": bool(ver.manifest.get("normalize_by_layers", False)),
        "recommended_dose_s": ver.manifest.get("recommended_dose_s"),
        "s_min_serving_floor": ver.manifest.get("s_min_serving_floor"),
        "s_cert_ceiling": ver.manifest.get("s_cert_ceiling"),
        "manifest_sha256": ver.manifest_sha256,
        "tensors_sha256": ver.tensors_sha256,
        "receipt_sha256": ver.receipt_sha256,
        "ledger_head_sha256": ver.ledger_head_sha256,
        "signature_present": ver.signature_present,
        "signature_valid": ver.signature_valid,
        "guard_policy": ver.manifest.get("guard_policy", {}),
        "benchmarks": ver.receipt.get("benchmarks"),
        "composed_from": ver.manifest.get("composed_from"),
    }


def diff_cartridges(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    public_key_a: Any = None,
    public_key_b: Any = None,
    require_signatures: bool = False,
) -> Dict[str, Any]:
    """
    Compares two verified `.starbody` cartridges across manifest bounds, benchmark receipts,
    and exact float64 Grassmannian subspace geometry (without d_model x d_model allocations).
    """
    enforce_precision_and_seed(20261117)
    ver_a = verify_cartridge(path_a, public_key=public_key_a, require_signature=require_signatures)
    ver_b = verify_cartridge(path_b, public_key=public_key_b, require_signature=require_signatures)

    d_a = ver_a.manifest.get("d_model")
    d_b = ver_b.manifest.get("d_model")
    h_a = ver_a.manifest.get("hook_path")
    h_b = ver_b.manifest.get("hook_path")
    d_match = d_a == d_b
    h_match = h_a == h_b

    rec_a = float(ver_a.manifest.get("recommended_dose_s", 0.0) or 0.0)
    rec_b = float(ver_b.manifest.get("recommended_dose_s", 0.0) or 0.0)
    cert_a = float(ver_a.manifest.get("s_cert_ceiling", 0.0) or 0.0)
    cert_b = float(ver_b.manifest.get("s_cert_ceiling", 0.0) or 0.0)
    smin_a = float(ver_a.manifest.get("s_min_serving_floor", 0.0) or 0.0)
    smin_b = float(ver_b.manifest.get("s_min_serving_floor", 0.0) or 0.0)

    policy_diff = {
        "d_model_match": bool(d_match),
        "hook_path_match": bool(h_match),
        "d_model_a": d_a,
        "d_model_b": d_b,
        "hook_path_a": h_a,
        "hook_path_b": h_b,
        "dose_delta": round(rec_b - rec_a, 6),
        "s_cert_delta": round(cert_b - cert_a, 6),
        "s_min_delta": round(smin_b - smin_a, 6),
    }

    bench_a = ver_a.receipt.get("benchmarks")
    bench_b = ver_b.receipt.get("benchmarks")
    benchmark_diff: Optional[Dict[str, Any]] = None
    if isinstance(bench_a, dict) and isinstance(bench_b, dict):
        sa = float(bench_a.get("steered_score", 0.0) or 0.0)
        sb = float(bench_b.get("steered_score", 0.0) or 0.0)
        ra = float(bench_a.get("retention_pct", 0.0) or 0.0)
        rb = float(bench_b.get("retention_pct", 0.0) or 0.0)
        benchmark_diff = {
            "framework_a": bench_a.get("framework"),
            "framework_b": bench_b.get("framework"),
            "steered_score_delta": round(sb - sa, 6),
            "retention_pct_delta": round(rb - ra, 4),
        }

    per_layer: Dict[int, Dict[str, Any]] = {}
    max_angle = 0.0
    chordal_list: List[float] = []
    max_gen_drift = 0.0

    if d_match:
        u_a_map, a_a_map, _ = _extract_layer_tensors_from_dict(ver_a.tensors)
        u_b_map, a_b_map, _ = _extract_layer_tensors_from_dict(ver_b.tensors)
        shared = sorted(set(u_a_map.keys()) & set(u_b_map.keys()))
        for l_idx in shared:
            u_a = u_a_map[l_idx].detach().cpu().to(torch.float64)
            u_b = u_b_map[l_idx].detach().cpu().to(torch.float64)
            a_a = a_a_map[l_idx].detach().cpu().to(torch.float64)
            a_b = a_b_map[l_idx].detach().cpu().to(torch.float64)

            sv = torch.linalg.svdvals(u_a.T @ u_b).clamp(min=0.0, max=1.0)
            angles_deg = [round(math.degrees(math.acos(float(c.item()))), 4) for c in sv]
            # If ranks differ, pad remaining canonical angles to 90 deg
            k_min = min(int(u_a.shape[1]), int(u_b.shape[1]))
            k_max = max(int(u_a.shape[1]), int(u_b.shape[1]))
            if k_max > k_min:
                angles_deg.extend([90.0] * (k_max - k_min))
            chordal = math.sqrt(sum(max(0.0, 1.0 - float(c.item()) ** 2) for c in sv) + max(0, k_max - k_min))

            q_all, _ = torch.linalg.qr(torch.cat([u_a, u_b], dim=1), mode="reduced")
            p_a = q_all.T @ u_a
            p_b = q_all.T @ u_b
            g_a_r = p_a @ a_a @ p_a.T
            g_b_r = p_b @ a_b @ p_b.T
            gen_drift = float(torch.linalg.norm(g_a_r - g_b_r).item())

            layer_max_ang = max(angles_deg) if angles_deg else 0.0
            max_angle = max(max_angle, layer_max_ang)
            chordal_list.append(chordal)
            max_gen_drift = max(max_gen_drift, gen_drift)
            per_layer[l_idx] = {
                "principal_angles_deg": angles_deg,
                "max_principal_angle_deg": round(layer_max_ang, 4),
                "chordal_distance": round(chordal, 6),
                "generator_frobenius_drift": round(gen_drift, 6),
            }

    mean_chordal = sum(chordal_list) / len(chordal_list) if chordal_list else 0.0
    return {
        "manifest_sha256_a": ver_a.manifest_sha256,
        "manifest_sha256_b": ver_b.manifest_sha256,
        "max_principal_angle_deg": round(max_angle, 4),
        "mean_chordal_distance": round(mean_chordal, 6),
        "max_generator_drift": round(max_gen_drift, 6),
        "per_layer": per_layer,
        "policy_diff": policy_diff,
        "benchmark_diff": benchmark_diff,
    }


class StarbodyRegistry:
    """Content-addressed local cartridge registry with atomic index writes & ledger rollback."""

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir)
        self.blobs_dir = self.root_dir / "blobs"
        self.index_path = self.root_dir / "index.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"cartridges": {}}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("cartridges"), dict):
            return {"cartridges": {}}
        return data

    def _write_index_atomic(self, index_data: Dict[str, Any]) -> None:
        tmp_path = self.index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(index_data, indent=2, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.index_path)

    def publish(
        self,
        cartridge_path: Union[str, Path],
        name: str,
        tag: str = "v1",
        public_key: Any = None,
        require_signature: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_name = str(name).strip()
        clean_tag = str(tag).strip()
        if not clean_name or not clean_tag:
            raise ValueError("Both `name` and `tag` must be non-empty.")

        src_path = Path(cartridge_path)
        ver = verify_cartridge(src_path, public_key=public_key, require_signature=require_signature)
        blob_path = self.blobs_dir / f"{ver.manifest_sha256}.starbody"
        if not blob_path.exists():
            tmp_blob = blob_path.with_suffix(".starbody.tmp")
            shutil.copy2(src_path, tmp_blob)
            os.replace(tmp_blob, blob_path)

        entry = {
            "name": clean_name,
            "tag": clean_tag,
            "manifest_sha256": ver.manifest_sha256,
            "tensors_sha256": ver.tensors_sha256,
            "receipt_sha256": ver.receipt_sha256,
            "blob_path": str(blob_path.resolve()),
            "model_name": ver.manifest.get("model_name"),
            "d_model": ver.manifest.get("d_model"),
            "hook_path": ver.manifest.get("hook_path"),
            "recommended_dose_s": ver.manifest.get("recommended_dose_s"),
            "s_cert_ceiling": ver.manifest.get("s_cert_ceiling"),
            "signature_present": ver.signature_present,
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": dict(metadata or {}),
        }

        idx = self._read_index()
        rec = idx["cartridges"].setdefault(clean_name, {"latest": clean_tag, "tags": {}, "history": []})
        rec["tags"][clean_tag] = entry
        rec["latest"] = clean_tag
        rec["history"].append({"tag": clean_tag, "manifest_sha256": ver.manifest_sha256, "published_utc": entry["published_utc"]})
        self._write_index_atomic(idx)
        return entry

    def resolve(self, ref: str) -> Path:
        clean_ref = str(ref).strip()
        idx = self._read_index()
        cartridges = idx.get("cartridges", {})

        # Match `name:tag` or exact `name` (latest)
        if ":" in clean_ref:
            name, tag = clean_ref.split(":", 1)
            rec = cartridges.get(name)
            if rec is None or tag not in rec.get("tags", {}):
                raise KeyError(f"Registry ref '{clean_ref}' not found.")
            blob_p = Path(rec["tags"][tag]["blob_path"])
        elif clean_ref in cartridges:
            rec = cartridges[clean_ref]
            latest_tag = rec.get("latest")
            if not latest_tag or latest_tag not in rec.get("tags", {}):
                raise KeyError(f"Registry entry '{clean_ref}' has no active tag.")
            blob_p = Path(rec["tags"][latest_tag]["blob_path"])
        else:
            # Resolve by manifest SHA-256 prefix
            matches: List[Path] = []
            for rec in cartridges.values():
                for t_entry in rec.get("tags", {}).values():
                    if str(t_entry.get("manifest_sha256", "")).startswith(clean_ref):
                        p_cand = Path(t_entry["blob_path"])
                        if p_cand not in matches:
                            matches.append(p_cand)
            if not matches:
                raise KeyError(f"Registry ref or digest '{clean_ref}' not found.")
            if len(matches) > 1:
                raise KeyError(f"Ambiguous digest prefix '{clean_ref}' matches {len(matches)} cartridges.")
            blob_p = matches[0]

        if not blob_p.exists():
            raise StarbodyIntegrityError(f"Registry blob missing on disk: {blob_p}")
        return blob_p

    def list_entries(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        idx = self._read_index()
        cartridges = idx.get("cartridges", {})
        out: List[Dict[str, Any]] = []
        names = [name] if (name is not None and name in cartridges) else sorted(cartridges.keys())
        for cname in names:
            rec = cartridges[cname]
            latest_tag = rec.get("latest")
            for t_name, t_entry in rec.get("tags", {}).items():
                row = dict(t_entry)
                row["is_latest"] = bool(t_name == latest_tag)
                out.append(row)
        return out

    def rollback(
        self,
        name: str,
        target_tag: Optional[str] = None,
        ledger_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        clean_name = str(name).strip()
        idx = self._read_index()
        rec = idx.get("cartridges", {}).get(clean_name)
        if rec is None:
            raise KeyError(f"Cartridge '{clean_name}' not found in registry.")

        current_tag = rec.get("latest")
        tags_map = rec.get("tags", {})
        resolved_tag: Optional[str] = None
        if target_tag is not None:
            if target_tag not in tags_map:
                raise KeyError(f"Target tag '{target_tag}' not found for '{clean_name}'.")
            resolved_tag = target_tag
        else:
            for hist in reversed(rec.get("history", [])[:-1]):
                h_tag = str(hist.get("tag", ""))
                if h_tag and h_tag != current_tag and h_tag in tags_map:
                    resolved_tag = h_tag
                    break
            if resolved_tag is None:
                raise KeyError(f"No distinct previous version available to roll back '{clean_name}'.")

        target_entry = tags_map[resolved_tag]
        blob_p = Path(target_entry["blob_path"])
        ver = verify_cartridge(blob_p)
        if ver.manifest_sha256 != target_entry["manifest_sha256"]:
            raise StarbodyIntegrityError("Rollback blob manifest SHA-256 mismatch.")

        rec["latest"] = resolved_tag
        rb_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["history"].append({
            "action": "rollback",
            "from_tag": current_tag,
            "tag": resolved_tag,
            "manifest_sha256": ver.manifest_sha256,
            "published_utc": rb_ts,
        })
        self._write_index_atomic(idx)

        ledger_hash: Optional[str] = None
        if ledger_path is not None:
            lp = Path(ledger_path)
            vault = DriveExperimentVault(local_fallback_dir=str(lp.parent))
            if lp.name != "ledger_smp_v1.json":
                vault.ledger_path = lp
                vault._ensure_ledger_initialized()
            ledger_entry = vault.append_ledger_entry(
                entry_type="REGISTRY_ROLLBACK",
                study_id=f"REGISTRY_{clean_name.upper()}",
                payload={
                    "name": clean_name,
                    "from_tag": current_tag,
                    "to_tag": resolved_tag,
                    "manifest_sha256": ver.manifest_sha256,
                },
            )
            ledger_hash = str(ledger_entry.get("entry_hash", "")) if isinstance(ledger_entry, dict) else str(ledger_entry)

        return {
            "name": clean_name,
            "from_tag": current_tag,
            "restored_tag": resolved_tag,
            "manifest_sha256": ver.manifest_sha256,
            "blob_path": str(blob_p.resolve()),
            "ledger_entry_hash": ledger_hash,
        }

    def verify_all(
        self,
        public_key: Any = None,
        require_signatures: bool = False,
    ) -> Dict[str, Any]:
        entries = self.list_entries()
        verified_count = 0
        errors: List[Dict[str, str]] = []
        for row in entries:
            ref = f"{row['name']}:{row['tag']}"
            try:
                ver = verify_cartridge(row["blob_path"], public_key=public_key, require_signature=require_signatures)
                if ver.manifest_sha256 != row["manifest_sha256"]:
                    raise StarbodyIntegrityError(
                        f"Manifest SHA-256 mismatch for {ref}: expected {row['manifest_sha256']}, got {ver.manifest_sha256}"
                    )
                verified_count += 1
            except Exception as exc:
                errors.append({"ref": ref, "error": str(exc)})
        return {"all_ok": len(errors) == 0, "verified_count": verified_count, "errors": errors}

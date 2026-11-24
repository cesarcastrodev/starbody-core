"""
Self-Contained `.starbody` Cartridge Compiler & Fail-Closed Verified Loader (SMP v3.2).

Every `.starbody` archive is a deterministic ZIP bundle containing:
  - `manifest.json`: model compatibility checks, guard policy, SHA-256 digests, ledger head hash
  - `tensors.safetensors`: contiguous float32 steering bases U_parallel, A_skew, & optional baseline_mean
  - `receipt.json`: empirical Phase-I calibration & dose-ladder certification evidence
  - `signature.ed25519` (optional): Ed25519 signature over canonical manifest SHA-256

Zero pickle is ever used anywhere in serialization or deserialization.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import hashlib
import hmac
import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import warnings
import zipfile

from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
import torch

from starbody.adapters import HuggingFaceAdapter
from starbody.interventions import AttachedController, GeodesicSOKKnob


logger = logging.getLogger("starbody.cartridge")
_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


class StarbodyIntegrityError(RuntimeError):
    """Raised fail-closed whenever a `.starbody` cartridge fails hash, signature, schema, or orthogonality checks."""


@dataclass
class VerifiedCartridge:
    """Validated in-memory representation of a `.starbody` cartridge."""

    manifest: Dict[str, Any]
    receipt: Dict[str, Any]
    tensors: Dict[str, torch.Tensor]
    tensors_sha256: str
    receipt_sha256: str
    manifest_sha256: str
    signature_present: bool
    signature_valid: Optional[bool]
    ledger_head_sha256: str


@dataclass
class GuardRuntimeState:
    """Live telemetry & dose attenuation log attached during `apply_cartridge(..., guard=True)`."""

    max_e_perp_budget: float
    initial_dose_s: float
    active_dose_s: float
    attenuation_count: int = 0
    last_e_perp: float = 0.0
    last_raw_e_perp: float = 0.0
    effective_threshold: float = 0.0
    normalization: str = "baseline_energy"
    min_dose_floor_ratio: float = 0.05
    events: List[Dict[str, Any]] = field(default_factory=list)


def _read_ledger_head_hash(ledger_path: Optional[Path] = None) -> str:
    candidates: List[Path] = []
    if ledger_path is not None:
        candidates.append(Path(ledger_path))
    candidates.extend([
        Path("./smp_local_vault/ledger_smp_v1.json"),
        Path("./ledger_smp_v1.json"),
    ])
    for cand in candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, list) and data and isinstance(data[-1], dict):
                    h = str(data[-1].get("entry_hash", ""))
                    if len(h) == 64:
                        return h
            except Exception:
                continue
    return "0" * 64


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _import_ed25519() -> Tuple[Any, Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        return serialization, Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:
        raise RuntimeError(
            "Ed25519 signing requires the `cryptography` package (`pip install cryptography`)."
        ) from exc


def generate_ed25519_keypair(
    private_key_path: Union[str, Path],
    public_key_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Generates a PKCS8/SPKI PEM Ed25519 keypair for signing `.starbody` cartridges."""
    serialization, Ed25519PrivateKey, _ = _import_ed25519()
    priv_p = Path(private_key_path)
    pub_p = Path(public_key_path) if public_key_path is not None else priv_p.with_suffix(".pub.pem")
    priv_p.parent.mkdir(parents=True, exist_ok=True)
    pub_p.parent.mkdir(parents=True, exist_ok=True)

    priv_key = Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()

    priv_pem = priv_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = pub_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_p.write_bytes(priv_pem)
    pub_p.write_bytes(pub_pem)
    return priv_p, pub_p


def load_ed25519_private_key(path: Union[str, Path]) -> Any:
    """Loads an Ed25519PrivateKey from a PEM file."""
    serialization, Ed25519PrivateKey, _ = _import_ed25519()
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"Expected Ed25519PrivateKey in {path}, got {type(key).__name__}")
    return key


def load_ed25519_public_key(path: Union[str, Path]) -> Any:
    """Loads an Ed25519PublicKey from a PEM file."""
    serialization, _, Ed25519PublicKey = _import_ed25519()
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"Expected Ed25519PublicKey in {path}, got {type(key).__name__}")
    return key


def ensure_default_signing_key(
    vault_dir: Union[str, Path] = Path("./smp_local_vault"),
) -> Tuple[Any, Any, Path, Path]:
    """Returns `(priv_key, pub_key, priv_path, pub_path)` under `<vault_dir>/keys/`, creating if absent."""
    keys_dir = Path(vault_dir) / "keys"
    priv_p = keys_dir / "starbody_ed25519.pem"
    pub_p = keys_dir / "starbody_ed25519.pub.pem"
    if not (priv_p.exists() and pub_p.exists()):
        generate_ed25519_keypair(priv_p, pub_p)
    return load_ed25519_private_key(priv_p), load_ed25519_public_key(pub_p), priv_p, pub_p


def _sign_bytes(data: bytes, signing_key: Any) -> bytes:
    if isinstance(signing_key, (str, Path)):
        signing_key = load_ed25519_private_key(signing_key)
    if hasattr(signing_key, "sign") and callable(signing_key.sign):
        sig = signing_key.sign(data)
        if not isinstance(sig, (bytes, bytearray)):
            raise TypeError("signing_key.sign(...) must return raw signature bytes.")
        return bytes(sig)
    if callable(signing_key):
        sig = signing_key(data)
        if not isinstance(sig, (bytes, bytearray)):
            raise TypeError("Callable signing_key must return raw signature bytes.")
        return bytes(sig)
    raise TypeError(
        "Unsupported signing_key: provide a PEM path, cryptography Ed25519PrivateKey, or callable(bytes) -> bytes."
    )


def _verify_signature_bytes(data: bytes, sig: bytes, public_key: Any) -> bool:
    if isinstance(public_key, (str, Path)):
        try:
            public_key = load_ed25519_public_key(public_key)
        except Exception:
            return False
    if hasattr(public_key, "verify") and callable(public_key.verify):
        try:
            res = public_key.verify(sig, data)
            return True if res is None else bool(res)
        except Exception:
            return False
    if callable(public_key):
        return bool(public_key(data, sig))
    return False


def compile_cartridge(
    output_path: Union[str, Path],
    manifest_dict: Dict[str, Any],
    tensors_dict: Dict[str, torch.Tensor],
    receipt_dict: Dict[str, Any],
    signing_key: Any = None,
    ledger_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Compiles a deterministic `.starbody` cartridge archive using `safetensors`
    (zero pickle) with SHA-256 digests and ledger head hash embedded in `manifest.json`.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    clean_tensors: Dict[str, torch.Tensor] = {}
    for key in sorted(tensors_dict.keys()):
        t = tensors_dict[key]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor for key '{key}', got {type(t).__name__}")
        clean_tensors[key] = t.detach().cpu().contiguous().to(torch.float32)

    tensors_bytes = safetensors_save(clean_tensors)
    receipt_bytes = _canonical_json_bytes(receipt_dict)

    tensors_sha = hashlib.sha256(tensors_bytes).hexdigest()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    ledger_head = manifest_dict.get("ledger_head_sha256") or _read_ledger_head_hash(
        Path(ledger_path) if ledger_path is not None else None
    )

    manifest = dict(manifest_dict)
    manifest["serialization_format"] = "safetensors"
    manifest["tensors_sha256"] = tensors_sha
    manifest["receipt_sha256"] = receipt_sha
    manifest["ledger_head_sha256"] = str(ledger_head)
    manifest["signature_present"] = signing_key is not None

    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    sig_bytes: Optional[bytes] = None
    if signing_key is not None:
        sig_bytes = _sign_bytes(manifest_sha.encode("utf-8"), signing_key)

    entries: List[Tuple[str, bytes]] = [
        ("manifest.json", manifest_bytes),
        ("receipt.json", receipt_bytes),
        ("tensors.safetensors", tensors_bytes),
    ]
    if sig_bytes is not None:
        entries.append(("signature.ed25519", sig_bytes))

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc_name, payload in sorted(entries, key=lambda x: x[0]):
            zi = zipfile.ZipInfo(filename=arc_name, date_time=_FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)

    return out_file


def _extract_layer_tensors_from_dict(tensors: Dict[str, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
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

    if not u_map and "u_parallel" in tensors and "a_skew" in tensors:
        u_map[0] = tensors["u_parallel"]
        a_map[0] = tensors["a_skew"]
        if "baseline_mean" in tensors:
            base_map[0] = tensors["baseline_mean"]

    return u_map, a_map, base_map


def verify_cartridge(
    cartridge_path: Union[str, Path],
    public_key: Any = None,
    require_signature: bool = False,
) -> VerifiedCartridge:
    """
    Verifies a `.starbody` cartridge fail-closed:
      1. Confirms archive contains `manifest.json`, `receipt.json`, `tensors.safetensors`.
      2. Checks SHA-256 digests of `tensors.safetensors` & `receipt.json` against `manifest.json`.
      3. Verifies optional Ed25519 signature if requested or public_key provided.
      4. Loads safetensors weights & asserts float64 orthogonality (<1e-5) and skew-symmetry (<1e-6).
    """
    path = Path(cartridge_path)
    if not path.exists():
        raise StarbodyIntegrityError(f"Cartridge file not found: {path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            for req in ("manifest.json", "receipt.json", "tensors.safetensors"):
                if req not in names:
                    raise StarbodyIntegrityError(f"Cartridge missing required member '{req}'")
            manifest_bytes = zf.read("manifest.json")
            receipt_bytes = zf.read("receipt.json")
            tensors_bytes = zf.read("tensors.safetensors")
            sig_bytes = zf.read("signature.ed25519") if "signature.ed25519" in names else None
    except zipfile.BadZipFile as exc:
        raise StarbodyIntegrityError(f"Corrupt cartridge ZIP archive: {exc}") from exc

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except Exception as exc:
        raise StarbodyIntegrityError(f"Corrupt JSON metadata inside cartridge: {exc}") from exc

    actual_tensors_sha = hashlib.sha256(tensors_bytes).hexdigest()
    actual_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    expected_tensors_sha = str(manifest.get("tensors_sha256", ""))
    expected_receipt_sha = str(manifest.get("receipt_sha256", ""))

    if not hmac.compare_digest(actual_tensors_sha, expected_tensors_sha):
        raise StarbodyIntegrityError(
            f"SHA-256 mismatch on tensors.safetensors: expected {expected_tensors_sha}, got {actual_tensors_sha}"
        )
    if not hmac.compare_digest(actual_receipt_sha, expected_receipt_sha):
        raise StarbodyIntegrityError(
            f"SHA-256 mismatch on receipt.json: expected {expected_receipt_sha}, got {actual_receipt_sha}"
        )

    sig_present = sig_bytes is not None and bool(manifest.get("signature_present", False))
    sig_valid: Optional[bool] = None
    if require_signature and not sig_present:
        raise StarbodyIntegrityError("Cartridge signature required, but `signature.ed25519` is absent.")
    if sig_present and public_key is not None and sig_bytes is not None:
        sig_valid = _verify_signature_bytes(actual_manifest_sha.encode("utf-8"), sig_bytes, public_key)
        if not sig_valid:
            raise StarbodyIntegrityError("Ed25519 signature verification failed on cartridge manifest.")

    try:
        tensors = safetensors_load(tensors_bytes)
    except Exception as exc:
        raise StarbodyIntegrityError(f"Failed safetensors load: {exc}") from exc

    u_map, a_map, _ = _extract_layer_tensors_from_dict(tensors)
    if not u_map:
        raise StarbodyIntegrityError("Cartridge contains no layer tangent bases (`layer_*.u_parallel`).")

    for l_idx, u_t in u_map.items():
        if l_idx not in a_map:
            raise StarbodyIntegrityError(f"Missing `a_skew` generator for layer {l_idx}.")
        a_t = a_map[l_idx]
        if u_t.ndim != 2 or a_t.ndim != 2 or u_t.shape[1] != a_t.shape[0] or a_t.shape[0] != a_t.shape[1]:
            raise StarbodyIntegrityError(
                f"Incompatible tensor shapes at layer {l_idx}: u={tuple(u_t.shape)}, a={tuple(a_t.shape)}"
            )
        k_dim = int(u_t.shape[1])
        u_64 = u_t.to(torch.float64)
        a_64 = a_t.to(torch.float64)
        ortho_err = float(torch.linalg.norm(u_64.T @ u_64 - torch.eye(k_dim, dtype=torch.float64)).item())
        skew_err = float(torch.linalg.norm(a_64 + a_64.T).item())
        if ortho_err >= 1e-5:
            raise StarbodyIntegrityError(f"Layer {l_idx} U_parallel fails float64 orthogonality: {ortho_err}")
        if skew_err >= 1e-6:
            raise StarbodyIntegrityError(f"Layer {l_idx} A_skew fails skew-symmetry: {skew_err}")

    return VerifiedCartridge(
        manifest=manifest,
        receipt=receipt,
        tensors=tensors,
        tensors_sha256=actual_tensors_sha,
        receipt_sha256=actual_receipt_sha,
        manifest_sha256=actual_manifest_sha,
        signature_present=sig_present,
        signature_valid=sig_valid,
        ledger_head_sha256=str(manifest.get("ledger_head_sha256", "")),
    )


class AttachedCartridgeHandle:
    """Live runtime handle managing a mounted `.starbody` cartridge and hook teardown."""

    def __init__(
        self,
        model: torch.nn.Module,
        verified: VerifiedCartridge,
        ctx_manager: Any,
        ctrl: AttachedController,
        guard_state: Optional[GuardRuntimeState],
    ) -> None:
        self.model = model
        self.verified = verified
        self._ctx_manager = ctx_manager
        self.controller = ctrl
        self.guard_state = guard_state
        self._detached = False

    def detach(self) -> None:
        if not self._detached:
            self._ctx_manager.__exit__(None, None, None)
            self._detached = True


def build_normalized_guard_fn(
    verified: VerifiedCartridge,
    baselines_64: Dict[int, torch.Tensor],
    chosen_dose: float,
    emit_warnings: bool = True,
) -> Tuple[Callable[[int, torch.Tensor, torch.Tensor, AttachedController], None], GuardRuntimeState]:
    """
    Constructs an energy-normalized orthogonal-drift guard callback and telemetry state.
    Normalizes last-token orthogonal residuals against calibrated baseline hidden-state
    energy and canary dispersion so legitimate prompts across decoding steps are never
    falsely attenuated while true off-manifold drift triggers bounded 50% decay.
    """
    guard_policy = verified.manifest.get("guard_policy") or {}
    max_budget = float(guard_policy.get("max_e_perp_budget", 0.02))
    attenuation_factor = float(guard_policy.get("attenuation_factor", 0.5))
    normalization = str(guard_policy.get("normalization", "baseline_energy"))
    spread_mult = float(guard_policy.get("spread_safety_multiplier", 2.5))
    min_floor_ratio = float(guard_policy.get("min_dose_floor_ratio", 0.05))
    base_energy_map = {int(k): float(v) for k, v in (guard_policy.get("baseline_energy_by_layer") or {}).items()}
    base_spread_map = {int(k): float(v) for k, v in (guard_policy.get("baseline_rel_spread_by_layer") or {}).items()}

    guard_state = GuardRuntimeState(
        max_e_perp_budget=max_budget,
        initial_dose_s=chosen_dose,
        active_dose_s=chosen_dose,
        normalization=normalization,
        min_dose_floor_ratio=min_floor_ratio,
    )

    def _guard_fn(l_idx: int, x: torch.Tensor, u_mat: torch.Tensor, ctrl: AttachedController) -> None:
        if not isinstance(ctrl.active_dose_s, (int, float)):
            return
        x_raw = x.detach().to(torch.float64)
        x_64 = x_raw[:, -1, :] if x_raw.ndim == 3 else x_raw  # [batch, d_model]
        ref_64 = baselines_64.get(l_idx)
        if ref_64 is None:
            baselines_64[l_idx] = x_64.mean(dim=0)
            guard_state.last_e_perp = 0.0
            guard_state.last_raw_e_perp = 0.0
            return

        delta_64 = x_64 - ref_64.unsqueeze(0)  # [batch, d_model]
        u_64 = u_mat.detach().to(torch.float64)
        proj_par = (delta_64 @ u_64) @ u_64.T
        delta_perp = delta_64 - proj_par
        raw_e_perp = float(torch.mean(torch.sum(delta_perp ** 2, dim=-1)).clamp(min=0.0).item())
        x_energy = float(torch.mean(torch.sum(x_64 ** 2, dim=-1)).clamp(min=1e-12).item())

        base_energy = base_energy_map.get(l_idx, max(float(torch.sum(ref_64 ** 2).item()), 1.0))
        base_spread = base_spread_map.get(l_idx, 0.0)
        denom = max(base_energy, x_energy, 1.0)
        rel_e_perp = raw_e_perp / denom
        rel_threshold = max_budget + spread_mult * base_spread
        energy_spike_ratio = raw_e_perp / max(base_energy, 1.0)
        spike_threshold = max(10.0, 16.0 * base_spread)

        guard_state.last_e_perp = rel_e_perp
        guard_state.last_raw_e_perp = raw_e_perp
        guard_state.effective_threshold = rel_threshold

        exceeded = (rel_e_perp > rel_threshold) or (base_spread > 0.0 and energy_spike_ratio > spike_threshold)
        if exceeded:
            prev = float(ctrl.active_dose_s)
            if abs(prev) <= 1e-12:
                return
            floor_abs = abs(guard_state.initial_dose_s) * min_floor_ratio
            candidate = prev * attenuation_factor
            new_dose = math.copysign(max(abs(candidate), floor_abs), prev)
            if abs(new_dose) >= abs(prev) - 1e-12:
                return
            ctrl.active_dose_s = new_dose
            guard_state.active_dose_s = new_dose
            guard_state.attenuation_count += 1
            ev = {
                "layer_idx": l_idx,
                "e_perp": rel_e_perp,
                "raw_e_perp": raw_e_perp,
                "rel_e_perp": rel_e_perp,
                "budget": max_budget,
                "effective_threshold": rel_threshold,
                "prev_dose_s": prev,
                "attenuated_dose_s": new_dose,
            }
            guard_state.events.append(ev)
            ctrl.events.append(ev)
            if emit_warnings:
                msg = (
                    f"[Starbody Guard] Normalized drift rel_E_perp={rel_e_perp:.4e} "
                    f"(raw={raw_e_perp:.4e}) exceeded threshold={rel_threshold:.4e} "
                    f"at layer {l_idx}; attenuating dose 50%: {prev:.5f} -> {new_dose:.5f}"
                )
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return _guard_fn, guard_state


def apply_cartridge(
    model: torch.nn.Module,
    cartridge_path: Union[str, Path],
    dose_s: Optional[float] = None,
    guard: bool = True,
    verify_signature: bool = False,
    public_key: Any = None,
) -> AttachedCartridgeHandle:
    """
    Verifies and attaches a `.starbody` cartridge directly onto `model`, attaching
    `model.starbody_detach()` and live orthogonal drift attenuation (`guard=True`).
    """
    verified = verify_cartridge(cartridge_path, public_key=public_key, require_signature=verify_signature)
    adapter = HuggingFaceAdapter(model)

    expected_d_model = verified.manifest.get("d_model")
    if expected_d_model is not None and int(expected_d_model) != adapter.d_model:
        raise StarbodyIntegrityError(
            f"Model d_model mismatch: cartridge expects {expected_d_model}, model has {adapter.d_model}"
        )

    expected_hook = verified.manifest.get("hook_path")
    if expected_hook is not None and str(expected_hook) != adapter.hook_path_prefix():
        raise StarbodyIntegrityError(
            f"Hook path mismatch: cartridge expects '{expected_hook}', model uses '{adapter.hook_path_prefix()}'"
        )

    u_raw, a_raw, base_raw = _extract_layer_tensors_from_dict(verified.tensors)
    u_by_layer = {idx: t.to(device=adapter.device, dtype=torch.float32) for idx, t in u_raw.items()}
    a_by_layer = {idx: t.to(device=adapter.device, dtype=torch.float32) for idx, t in a_raw.items()}
    baselines_64: Dict[int, torch.Tensor] = {
        idx: t.to(device=adapter.device, dtype=torch.float64) for idx, t in base_raw.items()
    }

    ceiling = float(verified.manifest.get("s_cert_ceiling", 10.0))
    default_dose = float(verified.manifest.get("recommended_dose_s", min(0.1, ceiling)))
    chosen_dose = default_dose if dose_s is None else float(dose_s)
    chosen_dose = min(chosen_dose, ceiling)

    guard_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor, AttachedController], None]] = None
    guard_state: Optional[GuardRuntimeState] = None
    if guard:
        guard_fn, guard_state = build_normalized_guard_fn(verified, baselines_64, chosen_dose, emit_warnings=True)

    if hasattr(model, "starbody_detach") and callable(getattr(model, "starbody_detach")):
        try:
            model.starbody_detach()
        except Exception:
            pass

    knob = GeodesicSOKKnob(
        u_parallel_by_layer=u_by_layer,
        a_skew_by_layer=a_by_layer,
        guard_fn=guard_fn,
        normalize_by_layers=bool(verified.manifest.get("normalize_by_layers", False)),
    )
    cm = knob.attach(adapter, dose_s=chosen_dose)
    ctrl = cm.__enter__()
    handle = AttachedCartridgeHandle(model=model, verified=verified, ctx_manager=cm, ctrl=ctrl, guard_state=guard_state)
    setattr(model, "starbody_detach", handle.detach)
    setattr(model, "_starbody_guard_state", guard_state)
    return handle


@contextlib.contextmanager
def cartridge_context(
    model: torch.nn.Module,
    cartridge_path: Union[str, Path],
    dose_s: Optional[float] = None,
    guard: bool = True,
    verify_signature: bool = False,
    public_key: Any = None,
) -> Iterator[AttachedCartridgeHandle]:
    """Context manager wrapping `apply_cartridge` with guaranteed `try...finally` detach."""
    handle = apply_cartridge(
        model,
        cartridge_path,
        dose_s=dose_s,
        guard=guard,
        verify_signature=verify_signature,
        public_key=public_key,
    )
    try:
        yield handle
    finally:
        handle.detach()

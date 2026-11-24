"""
Starbody Onboarding & `.starbody` Cartridge Architecture (SMP v3.2).

3-line developer quickstart:
    import starbody
    conn = starbody.connect(model)
    conn.compile("model.starbody")
"""

from __future__ import annotations

from starbody.adapters import (
    AgentProxyAdapter,
    BaseModelAdapter,
    HuggingFaceAdapter,
)
from starbody.cartridge import (
    AttachedCartridgeHandle,
    GuardRuntimeState,
    StarbodyIntegrityError,
    VerifiedCartridge,
    apply_cartridge,
    cartridge_context,
    compile_cartridge,
    ensure_default_signing_key,
    generate_ed25519_keypair,
    load_ed25519_private_key,
    load_ed25519_public_key,
    verify_cartridge,
)
from starbody.connect import (
    StarbodyConnection,
    connect,
    estimate_fp32_vram_mb,
    list_cached_hf_models,
)
from starbody.interventions import (
    AttachedController,
    GeodesicSOKKnob,
    InterventionKnob,
    LinearPatchKnob,
    LoRAAlphaKnob,
    QuantizationStepKnob,
)

try:
    from starbody_sdk import (
        StarbodyBifurcationWarning,
        certify,
        extract_tangent_basis_auto,
    )
except ImportError:  # pragma: no cover
    StarbodyBifurcationWarning = None  # type: ignore[assignment,misc]
    certify = None  # type: ignore[assignment]
    extract_tangent_basis_auto = None  # type: ignore[assignment]

__all__ = [
    "AgentProxyAdapter",
    "AttachedCartridgeHandle",
    "AttachedController",
    "BaseModelAdapter",
    "GeodesicSOKKnob",
    "GuardRuntimeState",
    "HuggingFaceAdapter",
    "InterventionKnob",
    "LinearPatchKnob",
    "LoRAAlphaKnob",
    "QuantizationStepKnob",
    "StarbodyBifurcationWarning",
    "StarbodyConnection",
    "StarbodyIntegrityError",
    "VerifiedCartridge",
    "apply_cartridge",
    "cartridge_context",
    "certify",
    "compile_cartridge",
    "connect",
    "ensure_default_signing_key",
    "estimate_fp32_vram_mb",
    "extract_tangent_basis_auto",
    "generate_ed25519_keypair",
    "list_cached_hf_models",
    "load_ed25519_private_key",
    "load_ed25519_public_key",
    "verify_cartridge",
]

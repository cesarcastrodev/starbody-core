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
from starbody.benchmarks import (
    DEFAULT_CORE_BENCHMARKS,
    evaluate_core_benchmarks,
    import_external_benchmark_receipt,
    run_lm_eval_benchmark,
)
from starbody.cartridge import (
    AttachedCartridgeHandle,
    GuardRuntimeState,
    StarbodyIntegrityError,
    VerifiedCartridge,
    apply_cartridge,
    build_normalized_guard_fn,
    cartridge_context,
    compile_cartridge,
    ensure_default_signing_key,
    generate_ed25519_keypair,
    load_ed25519_private_key,
    load_ed25519_public_key,
    verify_cartridge,
)
from starbody.compose import (
    ComposedCartridgeReport,
    compose,
    compose_cartridges,
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
from starbody.registry import (
    StarbodyRegistry,
    diff_cartridges,
    inspect_cartridge,
)
from starbody.serve import (
    StarbodyServingRuntime,
    create_serving_app,
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
    "ComposedCartridgeReport",
    "DEFAULT_CORE_BENCHMARKS",
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
    "StarbodyRegistry",
    "StarbodyServingRuntime",
    "VerifiedCartridge",
    "apply_cartridge",
    "build_normalized_guard_fn",
    "cartridge_context",
    "certify",
    "compile_cartridge",
    "compose",
    "compose_cartridges",
    "connect",
    "create_serving_app",
    "diff_cartridges",
    "ensure_default_signing_key",
    "estimate_fp32_vram_mb",
    "evaluate_core_benchmarks",
    "extract_tangent_basis_auto",
    "generate_ed25519_keypair",
    "import_external_benchmark_receipt",
    "inspect_cartridge",
    "list_cached_hf_models",
    "load_ed25519_private_key",
    "load_ed25519_public_key",
    "run_lm_eval_benchmark",
    "verify_cartridge",
]

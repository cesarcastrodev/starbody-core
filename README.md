# starbody-core — Synergetic Manifold Pharmacology (SMP v3.4)

Pure-emergent geometric metrology, 3-line developer onboarding (`starbody/`), universal model adapters & intervention knobs, multi-cartridge subspace composition, standardized capability benchmark receipts, OpenAI-compatible multi-tenant serving, and cryptographically verified Ed25519 `.starbody` cartridges.

---

## Protocol & Capabilities Overview (SMP v3.4 Sprint 2)

- **3-Line Developer Onboarding (`starbody.connect`):**
  ```python
  import starbody

  conn = starbody.connect(model, intervention="geodesic_so_k", target_pairs=pairs, canaries=canaries)
  cal = conn.calibrate()               # Phase-I gates: TF32 lock, sham floor, QR orthogonality, relative RMS s_min
  bench = conn.run_benchmarks()        # Standardized baseline vs. steered capability receipt (starbody_core_v1 / lm-eval)
  conn.compile("my_steer.starbody")    # Compiles safetensors + SHA-256 + Ed25519 signed cartridge
  ```
- **Scale-Invariant Relative `s_min` & Energy-Normalized Runtime Guard:**
  - `calibrate()` computes relative RMS quantization noise (`serving_floor_rel_rms = sqrt(srv_mse / mean(||x||^2))`) so `s_min_serving_floor` stays scale-invariant in radian units across architectures (`serving_floor_ok` requires `< 0.25`).
  - `build_normalized_guard_fn()` normalizes live orthogonal residuals against per-layer `baseline_energy_by_layer` and `baseline_rel_spread_by_layer`, eliminating false decode-step attenuation while preserving bounded 50% dose decay down to `min_dose_floor_ratio` (`0.05`) on true off-manifold drift.
- **Multi-Cartridge Subspace Orthogonalizer (`starbody/compose.py`, `starbody.compose`):**
  - Verifies two or more `.starbody` cartridges fail-closed, computes pairwise principal-angle overlap matrices in `float64`, enforces optional `max_interference_cos` collision guards, orthonormalizes the union subspace via thin SVD + reduced QR without allocating `d_model x d_model` matrices, projects weighted Lie-algebra generators `A_joint`, and asserts exact generator reconstruction bounds.
- **Standardized Benchmark Receipts (`starbody/benchmarks.py`):**
  - Built-in 12-item `starbody_core_v1` (`truthful_mc`, `reasoning_mc`, `safety_refusal`) with tensor-only fallback, live `lm-eval` (`lm_eval.simple_evaluate`) wrapper, and SHA-256-verified external JSON log importers for both `lm-eval` and `Inspect AI`.
- **OpenAI-Compatible Multi-Tenant Steering Proxy (`starbody/serve.py`):**
  - `StarbodyServingRuntime` and `create_serving_app()` serve `/health`, `/v1/models`, `/v1/chat/completions` (JSON + SSE `stream=True`, `model:cartridge` aliases, `X-Starbody-*` headers, `strict_bounds`), `/v1/completions`, and `/v1/starbody/batch_completions` (left-padded single-batch multi-tenant routing with bit-exact zero-dose identity).
- **Universal Model Adapters & Knobs (`starbody/adapters.py`, `starbody/interventions.py`):**
  - `HuggingFaceAdapter` (`model.model.layers`, `model.gpt_neox.layers`, `model.transformer.h`, `model.layers`), `AgentProxyAdapter` (Interactome Fiedler `lambda_2` / `v_50`), `GeodesicSOKKnob`, `LinearPatchKnob`, `QuantizationStepKnob`, and `LoRAAlphaKnob`.

---

## Hardware & Precision Invariants (RTX 3050 4 GB Local Standard)

All modules enforce strict determinism and numerical fidelity via `enforce_precision_and_seed()`:

- **Strict IEEE-754 `float32` Tensor Execution:** `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_float32_matmul_precision("highest")`.
- **High-Precision Accumulators:** QR decomposition, SVD, exposure sums ($E_\parallel, E_\perp$), orthogonal invariance metrics, normalized guard ratios, and lock-in susceptibility quadrature run in `float64` with `1e-12` epsilon clamping.
- **4 GB VRAM Guard:** Real HF models run in strict `float32` (`low_cpu_mem_usage=True`, `requires_grad_(False)`) with pre-flight VRAM budgeting (`<= 3072 MB` cap on NVIDIA GeForce RTX 3050 Laptop GPU; e.g., `HuggingFaceTB/SmolLM2-135M` peaks around `~537–551 MB` across calibration, composition, benchmarks, and serving).

---

## Verified Cryptographic Ledger Head (`ledger_smp_v1.json`)

- **Entries Verified:** `#1` through `#37` (`CHAIN_OK = True`)
- **Latest Head Entry:** `#37` (`SMP_V3_2_CI_GATE_HOPFIELD`)
- **Head SHA-256 Digest:** `dfcd5c0cc6a56e17d96da51f9722c3c8768cfa4d05bb67bf17ebb05b1aa8bdb8`

---

## Quickstart & CLI (`starbody_cli.py`)

```bash
# 1. Install CPU or CUDA PyTorch + editable package
pip install -e ".[dev]"

# 2. Run the Pre-Sealing Instrument Audit
python smp_core.py

# 3. Compose two signed cartridges with interference audit
python starbody_cli.py compose --cartridges a.starbody b.starbody --output composed.starbody

# 4. Run capability benchmark receipts or import lm-eval/Inspect AI logs
python starbody_cli.py eval --model sandbox --suite starbody_core_v1 --dose 0.10

# 5. Boot the OpenAI-compatible multi-tenant steering proxy
python starbody_cli.py serve --model HuggingFaceTB/SmolLM2-135M --cartridge joint=composed.starbody --port 8080 --local-only

# 6. Run automated tests
pytest -v
```

---

## License

Released under the **MIT License** (see `LICENSE`).

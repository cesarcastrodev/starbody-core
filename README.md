# starbody-core — Synergetic Manifold Pharmacology (SMP v3.5)

Pure-emergent geometric metrology, 3-line developer onboarding (`starbody/`), universal model adapters & intervention knobs, multi-cartridge subspace composition, content-addressed `.starbody` registry & Grassmannian diffs, standardized benchmark receipts, OpenAI-compatible serving proxy with Prometheus `/metrics` & atomic hot-swap/rollback, and pure-emergent multi-agent syncytium telemetry.

---

## Protocol & Capabilities Overview (SMP v3.5 Sprint 3)

- **3-Line Developer Onboarding (`starbody.connect`):**
  ```python
  import starbody

  conn = starbody.connect(model, intervention="geodesic_so_k", target_pairs=pairs, canaries=canaries)
  cal = conn.calibrate()               # Phase-I gates + baseline_energy/rel_spread guard calibration
  bench = conn.run_benchmarks()        # Standardized baseline vs. steered receipt (starbody_core_v1 / lm-eval)
  conn.compile("my_steer.starbody")    # Compiles safetensors + SHA-256 + Ed25519 signed cartridge
  ```
- **Content-Addressed Registry, Inspector & Grassmannian Diff (`starbody/registry.py`):**
  - `StarbodyRegistry`: atomic `blobs/<manifest_sha256>.starbody` storage + `os.replace` `index.json`, resolving `name:tag`, `name` (`latest`), or SHA-256 prefix, `verify_all()`, and hash-chained `REGISTRY_ROLLBACK` ledger sealing (`rollback()`).
  - `inspect_cartridge()` & `diff_cartridges()`: `float64` principal canonical angles (`degrees`), Grassmannian chordal distance, exact generator Frobenius drift, policy deltas, and benchmark deltas without allocating `d_model x d_model` matrices.
- **Serving Proxy Observability, Hot-Swap & Rollback (`starbody/serve.py`):**
  - `StarbodyServingRuntime` and `create_serving_app()` provide `GET /metrics` (Prometheus OpenMetrics text), `GET /v1/starbody/audit_log` (PII-safe `prompt_sha256_prefix`), atomic `POST /v1/starbody/cartridges/swap` (`swap_cartridge`), `POST /v1/starbody/cartridges/rollback`, `DELETE /v1/starbody/cartridges/{name}`, `/v1/chat/completions` (JSON + SSE), and `/v1/starbody/batch_completions`.
- **Pure-Emergent Multi-Agent Syncytium (`AgentProxyAdapter` in `starbody/adapters.py`):**
  - Evaluates multi-agent turn logs (`List[Dict[str, str]]`) through `InteractomeLaplacianTracker` to compute live Fiedler connectivity (`lambda_2`), `lambda2_ratio`, fracture risk (`NOMINAL`, `TREMOR_WARNING`, `FRACTURE_IMMINENT`), and exact emergent reversal barrier `consensus_v50_tau` via bounded bisection over `apply_counter_intervention_after_delay`.
- **Scale-Invariant Relative `s_min`, Energy-Normalized Guard & Multi-Cartridge Composition:**
  - Normalized runtime guard (`build_normalized_guard_fn`) scales residuals by calibrated per-layer `baseline_energy_by_layer` and canary spread, preventing false decode-step attenuation while clipping true off-manifold drift cleanly.
  - `starbody.compose()` (`compose_cartridges`) orthonormalizes union subspaces via thin SVD + reduced QR and projects weighted Lie-algebra generators in `float64`.

---

## Hardware & Precision Invariants (RTX 3050 4 GB Local Standard)

All modules enforce strict determinism and numerical fidelity via `enforce_precision_and_seed()`:

- **Strict IEEE-754 `float32` Tensor Execution:** `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_float32_matmul_precision("highest")`.
- **High-Precision Accumulators:** QR decomposition, SVD, exposure sums ($E_\parallel, E_\perp$), orthogonal invariance metrics, Grassmannian diffs, normalized guard ratios, and lock-in susceptibility quadrature run in `float64` with `1e-12` epsilon clamping.
- **4 GB VRAM Guard:** Real HF models run in strict `float32` (`low_cpu_mem_usage=True`, `requires_grad_(False)`) with pre-flight VRAM budgeting (`<= 3072 MB` cap on NVIDIA GeForce RTX 3050 Laptop GPU).

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

# 3. Inspect, diff & manage registry cartridges
python starbody_cli.py inspect my_steer.starbody
python starbody_cli.py diff a.starbody b.starbody
python starbody_cli.py registry publish --cartridge my_steer.starbody --name policy --tag v1
python starbody_cli.py registry verify

# 4. Compose, benchmark & serve OpenAI-compatible multi-tenant steering proxy
python starbody_cli.py compose --cartridges a.starbody b.starbody --output composed.starbody
python starbody_cli.py eval --model sandbox --suite starbody_core_v1 --dose 0.10
python starbody_cli.py serve --model HuggingFaceTB/SmolLM2-135M --cartridge joint=composed.starbody --port 8080 --local-only

# 5. Run automated tests
pytest -v
```

---

## License

Released under the **MIT License** (see `LICENSE`).

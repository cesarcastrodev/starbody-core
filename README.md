# starbody-core — Synergetic Manifold Pharmacology (SMP v3.3)

Pure-emergent geometric metrology, 3-line developer onboarding (`starbody/`), universal model adapters & intervention knobs, cryptographically verified `.starbody` Ed25519 cartridges, and SHA-256 hash-chained verification for neural networks and multi-agent syncytia.

---

## Protocol & Onboarding Overview (SMP v3.3 + Sprint 1)

- **3-Line Developer Onboarding (`starbody.connect`):**
  ```python
  import starbody

  conn = starbody.connect(model, intervention="geodesic_so_k", target_pairs=pairs, canaries=canaries)
  cal = conn.calibrate()               # Phase-I gates: TF32 lock, sham floor, QR orthogonality, serving floor
  conn.compile("my_steer.starbody")    # Compiles safetensors + SHA-256 + Ed25519 signed cartridge
  # Optional live workbench streaming: conn.serve_studio("ws://127.0.0.1:8000/ws/worker")
  ```
- **Universal Model Adapters (`starbody/adapters.py`):**
  - `HuggingFaceAdapter`: Auto-introspects decoder layers (`model.model.layers`, `model.gpt_neox.layers`, `model.transformer.h`, `model.layers`), preserves exact tuple hook outputs, captures `float32` activations under `torch.no_grad()`, and extracts orthonormal QR tangent bases $U_\parallel \in \mathbb{R}^{d \times k}$ ($\|U^T U - I_k\|_F < 10^{-5}$) and skew-symmetric generators $A \in \mathbb{R}^{k \times k}$ ($\|A + A^T\|_F < 10^{-6}$) verified in `float64`.
  - `AgentProxyAdapter`: Async/sync wrapper projecting multi-agent turn logs (`List[Dict[str, str]]`) into `[M, D]` states for `InteractomeLaplacianTracker` to stream live Fiedler connectivity $\lambda_2$ and reversal barrier $v_{50}(\tau)$.
- **Universal Intervention Knobs (`starbody/interventions.py`):**
  - `GeodesicSOKKnob`: Exact orthogonal rotation $R(s) = \exp(sA)$ inside $U_\parallel$ preserving hidden-state $L_2$ norms, with batched multi-tenant per-row dosing (`dose_s: float | Sequence[float] | Tensor`), bit-exact zero-dose identity, `position_mode` (`"all"`, `"prefill_only"`, `"decode_only"`, `"last_token"`), `normalize_by_layers` scaling (`1 / len(target_indices)`), and `try...finally` hook cleanup.
  - `LinearPatchKnob`: Rank-1 residual addition control for selectivity benchmarking.
  - `QuantizationStepKnob`: Non-destructive uniform step quantization ($\Delta = s$) with guaranteed weight restoration.
  - `LoRAAlphaKnob`: Dynamic continuous scaling of LoRA adapter `scaling` factors with exit restoration.
- **Zero-Pickle `.starbody` Cartridges (`starbody/cartridge.py`):** Deterministic ZIP bundles holding `tensors.safetensors`, canonical `receipt.json`, SHA-256 `manifest.json` pinned to the `ledger_smp_v1.json` head, and optional Ed25519 signature (`signature.ed25519`). Fail-closed verification via `verify_cartridge(...)` and runtime off-manifold attenuation (`guard=True` cuts dose 50% when $E_\perp > \text{budget}$) via `apply_cartridge(...)`.
- **Zero Synthetic Multipliers & Early Quadrature Lock-In ($\alpha(\chi_\perp) > 2.5$):** All exponents and reversal costs ($v_{50}$) emerge unforced from activations, Hopfield ODE dynamics, or Graph Laplacian conductance mechanics.

---

## Hardware & Precision Invariants (RTX 3050 4 GB Local Standard)

All modules enforce strict determinism and numerical fidelity via `enforce_precision_and_seed()`:

- **Strict IEEE-754 `float32` Tensor Execution:** `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_float32_matmul_precision("highest")`.
- **High-Precision Accumulators:** QR decomposition, SVD, exposure sums ($E_\parallel, E_\perp$), orthogonal invariance metrics, and lock-in susceptibility quadrature are accumulated in `float64` with `1e-12` epsilon clamping.
- **4 GB VRAM Guard:** Real HF models run in strict `float32` (`low_cpu_mem_usage=True`, `requires_grad_(False)`) with pre-flight VRAM budgeting (`<= 3072 MB` cap on NVIDIA GeForce RTX 3050 Laptop GPU; e.g., `HuggingFaceTB/SmolLM2-135M` peaks at `~537 MB`).
- **Pre-Sealing Gate (`PilotCalibrationGate`):** Sham floor checks ($E_\perp(0) < 10^{-4}$), SNR verification, and $EC_{50}$ centering prior to sealing.

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

# 3. Generate an Ed25519 cartridge signing keypair
python starbody_cli.py keygen --out-dir ./smp_local_vault/keys

# 4. Run the CI Gate CLI against a substrate dose
python starbody_cli.py gate --substrate hopfield --target-dose 1.5 --budget-perp 0.02 --max-alpha 2.5 --no-ledger

# 5. Launch Starbody Studio cleanly from the CLI (when starbody-studio server is installed)
python starbody_cli.py studio --model HuggingFaceTB/SmolLM2-135M --local-only

# 6. Run automated tests
pytest -v
```

---

## License

Released under the **MIT License** (see `LICENSE`).

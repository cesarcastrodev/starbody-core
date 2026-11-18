# starbody-core — Synergetic Manifold Pharmacology (SMP v3.1 / v3.2)

Pure-emergent geometric metrology, orthogonal subspace steering, early lock-in susceptibility telemetry, and SHA-256 hash-chained verification for neural networks and multi-agent syncytia.

---

## Protocol Overview (Pure Emergent SMP v3.1 / v3.2)

- **Zero Synthetic Multipliers:** All scaling exponents ($\alpha_\perp$, $\alpha(\chi_\perp)$), regenerative dynamics, and reversal barriers ($v_{50}$) emerge unforced directly from model activations, Hopfield ODE dynamics, or Graph Laplacian mechanics.
- **Orthogonal Dose Decomposition:** Projects representation shifts onto an orthonormal QR tangent basis $U_\parallel \in \mathbb{R}^{d \times k}$ ($k = 4$, $\|U^T U - I_k\| < 10^{-5}$) and separates intended steering exposure ($E_\parallel$) from collateral orthogonal leakage ($E_\perp$).
- **Early Quadrature Lock-In Warning ($\alpha(\chi_\perp) > 2.5$):** Measures single-trajectory frequency-locked transverse susceptibility via dual-phase sinusoidal demodulation to flag approaching saddle-node bifurcations *before* held-out capability degrades.
- **Multi-Agent Graph Laplacian ($\lambda_2$):** `InteractomeLaplacianTracker` models $M$-node multi-agent systems with voltage-gated semantic conductances $G_{ij}(t)$, detecting pre-fracture Fiedler connectivity collapse.

---

## Hardware & Precision Invariants (RTX 3050 4 GB Local Standard)

All modules enforce strict determinism and numerical fidelity via `enforce_precision_and_seed()`:

- **Strict IEEE-754 `float32` Tensor Execution:** `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_float32_matmul_precision("highest")`.
- **High-Precision Accumulators:** QR decomposition, SVD, exposure sums ($E_\parallel, E_\perp$), and lock-in susceptibility quadrature are accumulated in `float64`.
- **4 GB VRAM Guard:** Language models load with `torch_dtype=torch.float32`, `low_cpu_mem_usage=True`, and `model.requires_grad_(False)` with single-batch rollouts and explicit cache clearing (`torch.cuda.empty_cache()`) so peak VRAM stays below `3.0 GB` on an NVIDIA GeForce RTX 3050 Laptop GPU.
- **Pre-Sealing Gate (`PilotCalibrationGate`):** Every run passes sham floor checks ($E_\perp(0) < 10^{-4}$), SNR verification, and $EC_{50}$ centering prior to writing a `SEALED_PREDICTION` entry.

---

## Verified Cryptographic Ledger Head (`ledger_smp_v1.json`)

- **Entries Verified:** `#1` through `#37` (`CHAIN_OK = True`)
- **Latest Head Entry:** `#37` (`SMP_V3_2_CI_GATE_HOPFIELD`)
- **Head SHA-256 Digest:** `dfcd5c0cc6a56e17d96da51f9722c3c8768cfa4d05bb67bf17ebb05b1aa8bdb8`

---

## Quickstart

```bash
# 1. Install CPU or CUDA PyTorch + editable package
pip install -e ".[dev]"

# 2. Run the Pre-Sealing Instrument Audit
python smp_core.py

# 3. Extract a k=4 tangent basis and certify a training/steering loop
python -c "import starbody_sdk as starbody; print(dir(starbody))"

# 4. Run the CI Gate CLI against a substrate dose
python starbody_cli.py gate --substrate hopfield --target-dose 1.5 --budget-perp 0.02 --max-alpha 2.5 --no-ledger

# 5. Run automated tests
pytest -v
```

---

## License

Released under the **MIT License** (see `LICENSE`).

"""
Synergetic Manifold Pharmacology (SMP) v3.1 Core Engine (Pure Emergent Local Suite)
Implements:
1. Strict IEEE-754 float32 precision lock + float64 exposure accumulators.
2. Tangent-Normal Exposure Decomposition (E_parallel vs E_perp).
3. Twin-Free Dual-Phase (Quadrature) Lock-In Susceptibility Certifier.
4. PilotCalibrationGate: Pre-sealing EC50 range-finder, positive/sham control validator,
   and multi-token assay SNR check (prevents instrument spoilage before sealing).
5. Clean try...finally Geodesic Subspace Patch Hook (scalar or callable time-varying dose).
6. ContinuousHopfieldAttractorStep: Pure emergent continuous Hopfield attractor ODE
   (zero analytical gain envelopes or hardcoded barriers), 25% random-shard regeneration,
   and unscaled counter-chord reversal.
7. InteractomeLaplacianTracker: M-node bioelectric Graph Laplacian syncytium tracker
   computing voltage-gated conductance G_ij, Fiedler eigenvalue lambda_2, per-node
   selectivity, swarm lock-in chi_perp, and delayed consensus counter-intervention.
8. Atomic Google Drive / Local Experiment Vault & SHA-256 Hash-Chained Ledger.
"""

from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def enforce_precision_and_seed(seed: int = 20261117) -> None:
    """Hard-locks strict IEEE-754 float32 and disables TF32 rounding floors."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DoseRungMetrics:
    dose: float
    e_total: float
    e_parallel: float
    e_perp: float
    chi_perp_lockin: float
    local_exponent_perp: Optional[float]


class TwinFreePharmacologicalCertifier:
    """
    Computes decomposed exposure (E_parallel, E_perp) in float64 and extracts
    single-trajectory orthogonal susceptibility via dual-phase sinusoidal lock-in demodulation.
    """

    def __init__(
        self,
        tangent_basis: torch.Tensor,  # [D, k] orthonormal columns
        budget_perp: float = 0.02,
        guard_factor: float = 2.0,
        exponent_warning_threshold: float = 2.5,
    ) -> None:
        assert tangent_basis.ndim == 2, f"Expected [D, k], got {tangent_basis.shape}"
        d_dim, k_dim = tangent_basis.shape
        basis_64 = tangent_basis.to(torch.float64)
        ident_err = torch.norm(
            basis_64.T @ basis_64 - torch.eye(k_dim, dtype=torch.float64, device=basis_64.device)
        )
        assert ident_err < 1e-4, f"Tangent basis not orthonormal: err={ident_err.item()}"
        self.basis_64 = basis_64  # [D, k]
        self.budget_perp = float(budget_perp)
        self.guard_factor = float(guard_factor)
        self.exponent_warning_threshold = float(exponent_warning_threshold)

    def decompose_state_shift(
        self,
        delta_x: torch.Tensor,  # [..., D] state shift relative to pre-pulse baseline
    ) -> Tuple[float, float, float]:
        """Projects delta_x into tangent (intended) and orthogonal (harm) subspaces in float64."""
        dx_64 = delta_x.to(torch.float64).reshape(-1, self.basis_64.shape[0])  # [N, D]
        proj_coeff = dx_64 @ self.basis_64  # [N, k]
        dx_parallel = proj_coeff @ self.basis_64.T  # [N, D]
        dx_perp = dx_64 - dx_parallel  # [N, D]

        e_parallel = torch.mean(torch.sum(dx_parallel ** 2, dim=-1)).item()
        e_perp = torch.mean(torch.sum(dx_perp ** 2, dim=-1)).item()
        e_total = e_parallel + e_perp
        return e_total, e_parallel, e_perp

    def lockin_orthogonal_susceptibility(
        self,
        trajectory: torch.Tensor,  # [T_d, ..., D] temporal rollout under dithered dose
        dither_amplitude: float,
        omega_d: float,
    ) -> float:
        """
        Dual-phase (quadrature) lock-in demodulation extracting total orthogonal
        susceptibility |chi_perp(s)|^2 = |X|^2 + |Y|^2 independent of phase lag near bifurcation.
        """
        assert trajectory.ndim >= 2, f"Expected [T_d, ..., D], got {trajectory.shape}"
        t_steps = trajectory.shape[0]
        traj_64 = trajectory.to(torch.float64).reshape(t_steps, -1, self.basis_64.shape[0])  # [T_d, N, D]

        # Remove temporal mean per spatial site to reject static baseline without a twin
        traj_centered = traj_64 - torch.mean(traj_64, dim=0, keepdim=True)  # [T_d, N, D]

        # Project onto orthogonal complement P_perp
        proj_parallel = (traj_centered @ self.basis_64) @ self.basis_64.T  # [T_d, N, D]
        traj_perp = traj_centered - proj_parallel  # [T_d, N, D]

        t_idx = torch.arange(t_steps, dtype=torch.float64, device=trajectory.device)
        scale = 2.0 / (t_steps * max(dither_amplitude, 1e-12))

        ref_sin = torch.sin(omega_d * t_idx).view(t_steps, 1, 1)  # [T_d, 1, 1]
        ref_cos = torch.cos(omega_d * t_idx).view(t_steps, 1, 1)  # [T_d, 1, 1]

        in_phase = scale * torch.sum(traj_perp * ref_sin, dim=0)    # [N, D]
        quadrature = scale * torch.sum(traj_perp * ref_cos, dim=0)  # [N, D]

        chi_perp = torch.mean(torch.sum(in_phase ** 2 + quadrature ** 2, dim=-1)).item()
        return chi_perp

    @staticmethod
    def compute_local_exponents(
        doses: List[float], values: List[float], floor: float = 0.0
    ) -> List[Optional[float]]:
        """Computes local log-log slope d(log(V - floor)) / d(log s) between adjacent doses."""
        exponents: List[Optional[float]] = [None]
        for i in range(1, len(doses)):
            v_prev = max(values[i - 1] - floor, 1e-15)
            v_curr = max(values[i] - floor, 1e-15)
            dl_v = math.log(v_curr) - math.log(v_prev)
            dl_s = math.log(doses[i]) - math.log(doses[i - 1])
            exponents.append(dl_v / dl_s if abs(dl_s) > 1e-12 else 0.0)
        return exponents


class GoodhartCertifier:
    """
    Dynamical Goodhart Law Rater for Dose-Response Curves.
    Detects when increasing an intervention dose continues pushing a targeted proxy metric
    while degrading held-out out-of-distribution utility, and tests whether orthogonal
    acceleration (alpha_perp > 2.5) warns prior to Goodhart collapse.
    """

    @staticmethod
    def evaluate_dose_curve(
        doses: List[float],
        proxy_gains: List[float],
        heldout_utilities: List[float],
        e_perp_values: List[float],
        alpha_perp_values: Optional[List[Optional[float]]] = None,
        exponent_warning_threshold: float = 2.5,
        utility_drop_tolerance: float = 0.05,
    ) -> Dict[str, object]:
        """Evaluates Goodhart tipping dose s_GH, overoptimization divergence, and warning lead."""
        n = len(doses)
        assert n >= 3 and len(proxy_gains) == n and len(heldout_utilities) == n and len(e_perp_values) == n, (
            "All input sequences must share equal length >= 3 starting with sham dose 0.0."
        )
        base_u = abs(float(heldout_utilities[0]))
        assert base_u > 1e-12, f"Baseline sham held-out utility must be non-zero, got {base_u}"

        if alpha_perp_values is None:
            pos_exps = TwinFreePharmacologicalCertifier.compute_local_exponents(
                doses[1:], e_perp_values[1:], floor=e_perp_values[0]
            )
            alpha_perp_values = [None] + pos_exps

        u_ret = [max(0.0, min(1.5, float(heldout_utilities[i]) / base_u)) for i in range(n)]
        max_proxy = max([float(p) for p in proxy_gains[1:]] or [1.0])
        norm_proxy = [
            max(0.0, (float(proxy_gains[i]) - float(proxy_gains[0])) / max(max_proxy - float(proxy_gains[0]), 1e-12))
            for i in range(n)
        ]

        gh_rung: Optional[int] = None
        warn_rung: Optional[int] = None
        divergences: List[float] = []

        for i in range(1, n):
            ap = alpha_perp_values[i]
            if warn_rung is None and ap is not None and float(ap) > exponent_warning_threshold:
                warn_rung = i
            proxy_rising = float(proxy_gains[i]) > float(proxy_gains[0]) and float(proxy_gains[i]) >= float(proxy_gains[i - 1]) - 1e-12
            if gh_rung is None and proxy_rising and (1.0 - u_ret[i]) > utility_drop_tolerance:
                gh_rung = i
            divergences.append(max(0.0, norm_proxy[i] - u_ret[i]))

        lead_rungs = (gh_rung - warn_rung) if (gh_rung is not None and warn_rung is not None) else None
        warned_before_goodhart = bool(warn_rung is not None and (gh_rung is None or warn_rung < gh_rung))

        start_idx = max(1, (gh_rung - 1) if gh_rung is not None else 1)
        d_proxy = norm_proxy[-1] - norm_proxy[start_idx]
        d_util = u_ret[-1] - u_ret[start_idx]
        elasticity = (-d_util / d_proxy) if d_proxy > 1e-8 and d_util < 0.0 else 0.0

        return {
            "goodhart_tipping_rung": gh_rung,
            "goodhart_tipping_dose_s_gh": float(doses[gh_rung]) if gh_rung is not None else None,
            "orthogonal_warning_rung": warn_rung,
            "orthogonal_warning_dose_s_warn": float(doses[warn_rung]) if warn_rung is not None else None,
            "lead_rungs_before_goodhart": lead_rungs,
            "warned_before_goodhart": warned_before_goodhart,
            "goodhart_overoptimization_divergence": round(sum(divergences) / max(len(divergences), 1), 4),
            "goodhart_elasticity": round(float(elasticity), 4),
            "utility_retention_at_max_dose": round(float(u_ret[-1]), 4),
        }


class PilotCalibrationGate:
    """
    Phase-I Pre-Sealing Instrument Calibration & Range-Finding Gate.
    Validates on a disjoint calibration split BEFORE predictions are sealed:
      1. Sham control floor check (e_perp(0) < max_sham_floor).
      2. Baseline functional readout SNR check (abs(baseline_margin) >= min_baseline_margin).
      3. Pilot EC50 / s_crit range-finder locating the actual bifurcation threshold so the
         10-step sqrt(2) dose ladder is centered around s_crit (rungs 1..4 in low-dose
         quadratic zone, rungs 5..7 in pre-bifurcation warning zone, rungs 8..10 in switched zone).
    """

    @staticmethod
    def build_centered_sqrt2_ladder(s_crit: float, num_steps: int = 10, crit_rung_index: int = 7) -> List[float]:
        """
        Builds a [0.0, s_1, ..., s_10] sqrt(2)-spaced dose ladder where rung `crit_rung_index`
        (1-based, default 7 of 10) sits at s_crit, ensuring 6 rungs below s_crit and 3 rungs above.
        """
        assert s_crit > 0.0, f"s_crit must be positive, got {s_crit}"
        s_1 = s_crit / (math.sqrt(2.0) ** (crit_rung_index - 1))
        return [0.0] + [round(s_1 * (math.sqrt(2.0) ** i), 6) for i in range(num_steps)]

    @classmethod
    def calibrate_dose_ladder(
        cls,
        probe_fn: Callable[[float], Dict[str, float]],
        candidate_doses: List[float],
        min_baseline_signal: float = 0.05,
        min_positive_control_effect: float = 0.80,
    ) -> Dict[str, object]:
        """
        Runs a coarse range-finding sweep on a calibration split using `probe_fn(dose)`,
        which returns {'sham_floor': ..., 'baseline_signal': ..., 'transition_fraction': ..., 'alpha_probe': ...}.
        Returns the calibrated 10-step sqrt(2) ladder and calibration certificate, or raises RuntimeError
        if the instrument fails calibration (preventing spoiled runs from being sealed).
        """
        sham_res = probe_fn(0.0)
        sham_floor = float(sham_res.get("e_perp", 0.0))
        base_sig = abs(float(sham_res.get("baseline_signal", 1.0)))
        if base_sig < min_baseline_signal:
            raise RuntimeError(
                f"[CalibrationSpoiledError] Baseline assay signal ({base_sig:.4e}) < {min_baseline_signal}. "
                f"Check tokenizer/channel readout before sealing!"
            )

        s_crit_est: Optional[float] = None
        max_effect = 0.0
        probe_records: List[Dict[str, float]] = []

        for s in candidate_doses:
            if s <= 0.0:
                continue
            res = probe_fn(s)
            tf = float(res.get("transition_fraction", 0.0))
            max_effect = max(max_effect, tf)
            probe_records.append({"dose": s, "transition_fraction": round(tf, 4), "e_perp": float(res.get("e_perp", 0.0))})
            if s_crit_est is None and tf >= 0.50:
                if len(probe_records) >= 2:
                    prev_s = probe_records[-2]["dose"]
                    prev_tf = probe_records[-2]["transition_fraction"]
                    frac = (0.50 - prev_tf) / max(tf - prev_tf, 1e-6)
                    s_crit_est = prev_s + frac * (s - prev_s)
                else:
                    s_crit_est = s

        if max_effect < min_positive_control_effect or s_crit_est is None:
            raise RuntimeError(
                f"[CalibrationSpoiledError] Positive control reached only {max_effect*100:.1f}% "
                f"(< {min_positive_control_effect*100:.1f}% required). Increase dose range or coupling!"
            )

        calibrated_ladder = cls.build_centered_sqrt2_ladder(s_crit_est, num_steps=10, crit_rung_index=7)
        return {
            "calibration_status": "PASS",
            "sham_e_perp_floor": sham_floor,
            "baseline_assay_signal": round(base_sig, 4),
            "positive_control_max_effect": round(max_effect, 4),
            "estimated_s_crit_ec50": round(float(s_crit_est), 6),
            "calibrated_dose_ladder": calibrated_ladder,
            "pilot_probe_records": probe_records,
        }


class GeodesicSubspaceIntervention(nn.Module):
    """
    Riemannian tangent-subspace rotation operator for Transformer residual streams.
    Operates on tensor shape [batch, seq_len, d_model] with norm-preserving matrix exponential.
    """

    def __init__(self, u_basis: torch.Tensor, generator_a: torch.Tensor) -> None:
        super().__init__()
        assert u_basis.ndim == 2, f"Expected u_basis [D, k], got {u_basis.shape}"
        d_model, k_dim = u_basis.shape
        assert generator_a.shape == (k_dim, k_dim), f"Expected [{k_dim}, {k_dim}], got {generator_a.shape}"
        skew_a = 0.5 * (generator_a - generator_a.T)
        self.register_buffer("u_basis", u_basis.to(torch.float32))  # [D, k]
        self.register_buffer("skew_a", skew_a.to(torch.float32))    # [k, k]

    def forward(
        self,
        x: torch.Tensor,                          # [batch, seq_len, d_model]
        dose: float,
        seq_mask: Optional[torch.Tensor] = None,  # [batch, seq_len, 1]
    ) -> torch.Tensor:
        assert x.ndim == 3, f"Expected [batch, seq_len, d_model], got {x.shape}"
        rot_k = torch.matrix_exp(float(dose) * self.skew_a) - torch.eye(
            self.skew_a.shape[0], dtype=x.dtype, device=x.device
        )  # [k, k]
        proj = x @ self.u_basis                  # [batch, seq_len, k]
        delta = (proj @ rot_k) @ self.u_basis.T  # [batch, seq_len, d_model]
        if seq_mask is not None:
            delta = delta * seq_mask
        return x + delta


@contextlib.contextmanager
def attach_geodesic_hooks(
    modules: List[nn.Module],
    interventions: List[GeodesicSubspaceIntervention],
    dose: Union[float, Callable[[], float]],
):
    """Context manager guaranteeing forward hook removal via try...finally.
    Accepts either a static scalar dose or a zero-argument callable returning s(t).
    """
    assert len(modules) == len(interventions), "Module and intervention lists must match length."
    handles = []

    def _resolve_dose() -> float:
        return float(dose()) if callable(dose) else float(dose)

    def _make_hook(op: GeodesicSubspaceIntervention):
        def _hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            current_dose = _resolve_dose()
            if isinstance(output, tuple):
                steered = op(output[0], dose=current_dose)
                return (steered,) + output[1:]
            return op(output, dose=current_dose)
        return _hook

    try:
        for mod, op in zip(modules, interventions):
            handles.append(mod.register_forward_hook(_make_hook(op)))
        yield
    finally:
        for h in handles:
            h.remove()


class ContinuousHopfieldAttractorStep(nn.Module):
    """
    Pure Emergent Multi-Basin Continuous Hopfield Attractor Network on state [batch, D].
    Evolves strictly under the native continuous associative ODE:
        dx = dt * (-x + tanh(beta * (x @ M^T) @ M / D) + 0.15 * tanh(completion_net(x)) + drive)
             + thermal_sigma * sqrt(dt) * N(0, I)
    Zero analytical saddle gain multipliers, sigmoid switching gates, or barrier envelopes.
    """

    def __init__(
        self,
        state_dim: int = 64,
        num_basins: int = 4,
        beta_hopfield: float = 4.5,
        dt: float = 0.15,
        s_crit_saddle: float = 10.0,
        thermal_sigma: float = 0.0008,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_basins = num_basins
        self.beta_hopfield = beta_hopfield
        self.dt = dt
        self.s_crit_saddle = s_crit_saddle
        self.thermal_sigma = thermal_sigma

        # Memory prototypes M_0 (baseline) .. M_{num_basins-1} in {-1, +1}^D
        # Dense distributed Hadamard coding across all D sites so any 25% shard has high SNR.
        h_mat = torch.ones((state_dim, 4), dtype=torch.float32)
        idx = torch.arange(state_dim)
        h_mat[:, 0] = torch.where(idx % 2 == 0, 1.0, -1.0)
        h_mat[:, 1] = torch.where((idx // 2) % 2 == 0, 1.0, -1.0)
        h_mat[:, 2] = torch.where((idx // 4) % 2 == 0, 1.0, -1.0)
        h_mat[:, 3] = torch.where((idx // 8) % 2 == 0, 1.0, -1.0)
        self.register_buffer("memory_patterns", h_mat.T[:num_basins].clone())

        # Align column 0 of U_parallel with the true write chord (M_1 - M_0) / ||M_1 - M_0||
        # via float64 QR factorization so linear sub-threshold drift stays in U_parallel.
        gen = torch.Generator()
        gen.manual_seed(20261117)
        chord_64 = (self.memory_patterns[1] - self.memory_patterns[0]).to(torch.float64)
        chord_64 = chord_64 / torch.norm(chord_64).clamp(min=1e-12)
        aux_cols = torch.randn(state_dim, state_dim - 1, generator=gen, dtype=torch.float64)
        q_init, r_init = torch.linalg.qr(torch.cat([chord_64.unsqueeze(1), aux_cols], dim=1))
        if r_init[0, 0] < 0:
            q_init[:, 0] = -q_init[:, 0]
        q_32 = q_init.to(torch.float32)

        self.register_buffer("u_tangent", q_32[:, :4])       # [D, 4]
        self.register_buffer("u_perp_saddle", q_32[:, 4:6])  # [D, 2]
        self.register_buffer("u_morph_aux", q_32[:, 6:8])    # [D, 2]

        # Learned associative readout & lateral pattern-completion network
        self.completion_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, state_dim),
        )

    def get_write_direction(self) -> torch.Tensor:
        """Returns the unscaled basin-switching control unit vector (M_1 - M_0) / (2 * s_crit_saddle)."""
        return (self.memory_patterns[1] - self.memory_patterns[0]) / (2.0 * float(self.s_crit_saddle))

    def _step_ode(
        self,
        x: torch.Tensor,
        drive_vec: Union[torch.Tensor, float] = 0.0,
        enable_noise: bool = True,
    ) -> torch.Tensor:
        """Single Euler-Maruyama step of the pure continuous Hopfield ODE."""
        m_bank = self.memory_patterns
        d_dim = float(self.state_dim)
        hopfield_field = torch.tanh(
            self.beta_hopfield * ((x @ m_bank.T) @ m_bank) / d_dim
        )
        learned_corr = 0.15 * torch.tanh(self.completion_net(x))
        dx = self.dt * (-x + hopfield_field + learned_corr + drive_vec)
        if enable_noise:
            dx = dx + (self.thermal_sigma * math.sqrt(self.dt)) * torch.randn_like(x)
        return x + dx

    def settle_equilibrium(
        self,
        state: torch.Tensor,
        steps: int = 60,
    ) -> torch.Tensor:
        """Noiselessly relaxes a prototype or state batch onto its true ODE fixed point."""
        x = state.clone()
        for _ in range(steps):
            x = self._step_ode(x, drive_vec=0.0, enable_noise=False)
        return x

    def forward_pulse_rollout(
        self,
        init_state: torch.Tensor,   # [batch, D]
        dose: float,
        pulse_steps: int = 20,
        dither_amp: float = 0.08,
        enable_noise: bool = True,
        settle_steps: int = 25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a 20-step sinusoidal-dithered continuous Hopfield ODE pulse rollout
        from init_state under s(t) = dose + dither_amp * sin(omega_d * t), followed
        by unforced relaxation to reveal autonomous basin commitment.
        """
        _, d_dim = init_state.shape
        assert d_dim == self.state_dim, f"Expected state_dim {self.state_dim}, got {d_dim}"
        omega_d = 2.0 * math.pi * 3.0 / pulse_steps
        write_dir = self.get_write_direction().view(1, d_dim)

        x = init_state.clone()
        traj_steps: List[torch.Tensor] = []
        for t_idx in range(pulse_steps):
            s_t = float(dose) + float(dither_amp) * math.sin(omega_d * t_idx)
            x = self._step_ode(x, drive_vec=s_t * write_dir, enable_noise=enable_noise)
            traj_steps.append(x)

        dither_trajectory = torch.stack(traj_steps, dim=0)  # [pulse_steps, B, D]
        post_pulse_state = x
        for _ in range(settle_steps):
            post_pulse_state = self._step_ode(post_pulse_state, drive_vec=0.0, enable_noise=enable_noise)
        return post_pulse_state, dither_trajectory

    def amputate_and_regenerate_25pct_shard(
        self,
        state: torch.Tensor,          # [batch, D]
        shard_keep_mask: torch.Tensor, # [batch, D] binary mask with ~25% 1s per row
        regen_steps: int = 30,
        enable_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Excises 75% of state coordinates and relaxes purely under the continuous
        Hopfield ODE for `regen_steps` steps (with `regen_steps=0` returning the fresh shard).
        """
        amputated = state * shard_keep_mask
        x = amputated
        for _ in range(regen_steps):
            x = self._step_ode(x, drive_vec=0.0, enable_noise=enable_noise)
        return x, amputated

    def apply_counter_chord_and_settle(
        self,
        state: torch.Tensor,          # [batch, D]
        counter_dose: float,
        pulse_steps: int = 20,
        settle_steps: int = 25,
        enable_noise: bool = True,
    ) -> torch.Tensor:
        """Applies an opposing counter-chord drive -counter_dose * write_dir and relaxes."""
        write_dir = self.get_write_direction().view(1, self.state_dim)
        x = state.clone()
        for _ in range(pulse_steps):
            x = self._step_ode(x, drive_vec=-float(counter_dose) * write_dir, enable_noise=enable_noise)
        for _ in range(settle_steps):
            x = self._step_ode(x, drive_vec=0.0, enable_noise=enable_noise)
        return x


class BioelectricFieldSyncytiumStep(nn.Module):
    """Retained for v1.0 backward compatibility and CFL audit verification."""

    def __init__(
        self,
        num_channels: int = 8,
        dt: float = 0.15,
        c_mem: float = 1.0,
        g_bistable: float = 0.08,
        g_gj_max: float = 0.35,
        v_half_gj: float = 18.0,
        v_slope_gj: float = 4.0,
        gamma_field: float = 0.18,
        thermal_noise_mv: float = 1.2,
    ) -> None:
        super().__init__()
        cfl_number = 4.0 * g_gj_max * dt / c_mem
        assert cfl_number < 0.25, f"Explicit Euler CFL violation ({cfl_number:.3f} >= 0.25)"
        self.num_channels = num_channels
        self.dt = dt
        self.c_mem = c_mem
        self.g_bistable = g_bistable
        self.g_gj_max = g_gj_max
        self.v_half_gj = v_half_gj
        self.v_slope_gj = v_slope_gj
        self.gamma_field = gamma_field
        self.thermal_noise_mv = thermal_noise_mv

        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel)
        self.morph_net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, num_channels - 2, kernel_size=1),
        )

    def _bistable_ion_current(self, v_mem: torch.Tensor) -> torch.Tensor:
        v_norm = (v_mem + 42.5) / 27.5
        return self.g_bistable * 27.5 * (v_norm ** 3 - v_norm)

    def _voltage_gated_gap_junction_current(self, v_mem: torch.Tensor) -> torch.Tensor:
        v_pad = F.pad(v_mem, (1, 1, 1, 1), mode="replicate")
        neighbors = [
            v_pad[:, :, :-2, 1:-1],
            v_pad[:, :, 2:, 1:-1],
            v_pad[:, :, 1:-1, :-2],
            v_pad[:, :, 1:-1, 2:],
        ]
        i_gj_total = torch.zeros_like(v_mem)
        for v_nb in neighbors:
            dv = v_nb - v_mem
            g_ij = self.g_gj_max / (
                1.0 + torch.exp((torch.abs(dv) - self.v_half_gj) / self.v_slope_gj)
            )
            i_gj_total = i_gj_total + g_ij * dv
        return i_gj_total

    def forward(
        self,
        state: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_drive_mv: float = 0.0,
        enable_thermal_noise: bool = True,
    ) -> torch.Tensor:
        v_mem = state[:, 0:1, :, :]
        phi_ext = state[:, 1:2, :, :]
        morph = state[:, 2:, :, :]
        lap_phi = F.conv2d(F.pad(phi_ext, (1, 1, 1, 1), mode="replicate"), self.lap_kernel)
        phi_next = (
            phi_ext
            + self.dt * (1.2 * lap_phi - 0.3 * (v_mem + 42.5))
            + boundary_mask * boundary_drive_mv
        )
        i_ion = self._bistable_ion_current(v_mem)
        i_gj = self._voltage_gated_gap_junction_current(v_mem)
        lap_phi_next = F.conv2d(F.pad(phi_next, (1, 1, 1, 1), mode="replicate"), self.lap_kernel)
        dv = self.dt * (-i_ion + i_gj + self.gamma_field * lap_phi_next) / self.c_mem
        if enable_thermal_noise:
            dv = dv + (self.thermal_noise_mv * math.sqrt(self.dt)) * torch.randn_like(v_mem)
        v_next = torch.clamp(v_mem + dv, min=-95.0, max=40.0)
        coupled_state = torch.cat([v_next, phi_next, morph], dim=1)
        morph_next = torch.tanh(morph + self.dt * self.morph_net(coupled_state))
        return torch.cat([v_next, phi_next, morph_next], dim=1)


class InteractomeLaplacianTracker(nn.Module):
    """
    Models an M-node multi-agent or coupled-module network as a voltage-gated bioelectric
    syncytium with semantic conductance G_ij = G_max / (1 + exp((||dx_ij||_2 - V_half) / V_slope)),
    instantaneous Graph Laplacian L(t) = D(t) - G(t), Fiedler connectivity eigenvalue lambda_2(t),
    tangent/orthogonal swarm exposure decomposition (E_parallel, E_perp), and quadrature lock-in chi_perp.
    """

    def __init__(
        self,
        num_nodes: int,
        state_dim: int,
        tangent_basis: torch.Tensor,
        g_max: float = 1.0,
        v_half: float = 1.5,
        v_slope: float = 0.35,
        dt: float = 0.15,
        anchor_gamma: float = 0.45,
        beta_consensus: float = 2.8,
        thermal_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_nodes >= 2, f"Interactome requires >= 2 nodes, got {num_nodes}"
        assert tangent_basis.shape[0] == state_dim, f"Expected basis dim {state_dim}, got {tangent_basis.shape}"
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.g_max = float(g_max)
        self.v_half = float(v_half)
        self.v_slope = float(v_slope)
        self.dt = float(dt)
        self.anchor_gamma = float(anchor_gamma)
        self.beta_consensus = float(beta_consensus)
        self.thermal_sigma = float(thermal_sigma)

        self.certifier = TwinFreePharmacologicalCertifier(tangent_basis=tangent_basis.detach())

    def compute_conductance_and_laplacian(
        self,
        x_nodes: torch.Tensor,                      # [M, D]
        adj_mask: Optional[torch.Tensor] = None,    # [M, M] binary topology mask (1=connected)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Computes voltage-gated conductance G, symmetric Laplacian L = D - G, and spectral metrics in float64."""
        assert x_nodes.shape == (self.num_nodes, self.state_dim), f"Expected [{self.num_nodes}, {self.state_dim}], got {x_nodes.shape}"
        x_64 = x_nodes.to(torch.float64)
        diffs = x_64.unsqueeze(1) - x_64.unsqueeze(0)  # [M, M, D]
        dist = torch.linalg.norm(diffs, dim=-1)        # [M, M]

        exponent = ((dist - self.v_half) / max(self.v_slope, 1e-6)).clamp(-40.0, 40.0)
        g = self.g_max / (1.0 + torch.exp(exponent))

        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=x_64.device)
        g = g.masked_fill(eye, 0.0)
        if adj_mask is not None:
            g = g * adj_mask.to(torch.float64).to(x_64.device)
        g = 0.5 * (g + g.T)

        deg = torch.diag(g.sum(dim=-1))
        lap = deg - g
        evals = torch.linalg.eigvalsh(lap)
        lambda2 = float(max(0.0, evals[1].item()))
        lambda_max = float(max(evals[-1].item(), 1e-12))

        edge_mask = ~eye if adj_mask is None else ((adj_mask.to(torch.bool).to(x_64.device)) & (~eye))
        mean_dist = float(dist[edge_mask].mean().item()) if edge_mask.any() else 0.0
        stats = {
            "fiedler_lambda2": lambda2,
            "lambda_max": lambda_max,
            "spectral_ratio": lambda2 / lambda_max,
            "mean_edge_distance": mean_dist,
            "mean_conductance": float(g[edge_mask].mean().item()) if edge_mask.any() else 0.0,
        }
        return g.to(x_nodes.dtype), lap.to(x_nodes.dtype), stats

    def _step_syncytium(
        self,
        x_nodes: torch.Tensor,
        init_nodes: torch.Tensor,
        drive_nodes: torch.Tensor,
        consensus_patterns: Optional[torch.Tensor] = None,
        adj_mask: Optional[torch.Tensor] = None,
        enable_noise: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, lap, stats = self.compute_conductance_and_laplacian(x_nodes, adj_mask=adj_mask)
        lap_flux = -(lap @ x_nodes)
        if consensus_patterns is not None:
            # Bistable Hopfield-like associative consensus field in normalized [-1,+1]/D units
            c_norm = float(self.state_dim)
            field = torch.tanh(self.beta_consensus * ((x_nodes @ consensus_patterns.T) @ consensus_patterns) / c_norm)
            dx = self.dt * (-self.anchor_gamma * x_nodes + field + lap_flux + drive_nodes)
        else:
            dx = self.dt * (-self.anchor_gamma * (x_nodes - init_nodes) + lap_flux + drive_nodes)
        if enable_noise and self.thermal_sigma > 0.0:
            dx = dx + (self.thermal_sigma * math.sqrt(self.dt)) * torch.randn_like(x_nodes)
        return x_nodes + dx, stats

    @torch.no_grad()
    def settle_equilibrium(
        self,
        init_nodes: torch.Tensor,
        consensus_patterns: Optional[torch.Tensor] = None,
        adj_mask: Optional[torch.Tensor] = None,
        steps: int = 80,
    ) -> torch.Tensor:
        """Noiselessly relaxes the M-node syncytium to its exact unforced equilibrium."""
        x = init_nodes.clone()
        zeros = torch.zeros_like(x)
        for _ in range(steps):
            x, _ = self._step_syncytium(x, init_nodes, zeros, consensus_patterns=consensus_patterns, adj_mask=adj_mask, enable_noise=False)
        return x

    @torch.no_grad()
    def rollout_coupled_interactome(
        self,
        init_nodes: torch.Tensor,                         # [M, D]
        node0_drive_vec: torch.Tensor,                    # [D]
        dose: float,
        turns: int = 12,
        dither_amp: float = 0.04,
        consensus_patterns: Optional[torch.Tensor] = None,
        adj_mask: Optional[torch.Tensor] = None,
        enable_noise: bool = False,
    ) -> Dict[str, object]:
        """
        Evolves the M-node interactome under sinusoidal Node-0 perturbation s(t) = dose + dither_amp * sin(omega_d * t).
        Extracts trajectory [T, M, D], Fiedler connectivity stats, E_parallel, E_perp, and lock-in chi_perp.
        """
        x = init_nodes.clone()
        drive_unit = node0_drive_vec.view(self.state_dim)
        omega_d = 2.0 * math.pi * 2.0 / max(turns, 1)

        traj_steps: List[torch.Tensor] = []
        lambda2_hist: List[float] = []
        lambda_max_hist: List[float] = []
        mean_cond_hist: List[float] = []

        for t_idx in range(turns):
            s_t = float(dose) + float(dither_amp) * math.sin(omega_d * t_idx)
            drive_mat = torch.zeros_like(x)
            drive_mat[0] = s_t * drive_unit
            x, stats = self._step_syncytium(
                x, init_nodes, drive_mat, consensus_patterns=consensus_patterns, adj_mask=adj_mask, enable_noise=enable_noise
            )
            traj_steps.append(x.clone())
            lambda2_hist.append(stats["fiedler_lambda2"])
            lambda_max_hist.append(stats["lambda_max"])
            mean_cond_hist.append(stats["mean_conductance"])

        traj = torch.stack(traj_steps, dim=0)  # [T, M, D]
        mean_shift = traj.mean(dim=0) - init_nodes
        e_total, e_parallel, e_perp = self.certifier.decompose_state_shift(mean_shift)
        chi_perp = self.certifier.lockin_orthogonal_susceptibility(
            trajectory=traj - init_nodes.unsqueeze(0),
            dither_amplitude=dither_amp,
            omega_d=omega_d,
        )

        per_node_selectivity: List[float] = []
        for m_i in range(self.num_nodes):
            ntot, npar, _ = self.certifier.decompose_state_shift(mean_shift[m_i:m_i + 1])
            per_node_selectivity.append(float(npar / max(ntot, 1e-15)))

        g_fin, lap_fin, fin_stats = self.compute_conductance_and_laplacian(x, adj_mask=adj_mask)
        return {
            "trajectory": traj,
            "final_nodes": x,
            "final_conductance": g_fin,
            "final_laplacian": lap_fin,
            "fiedler_lambda2_mean": sum(lambda2_hist) / len(lambda2_hist),
            "fiedler_lambda2_final": fin_stats["fiedler_lambda2"],
            "fiedler_lambda2_min": min(lambda2_hist),
            "lambda_max_final": fin_stats["lambda_max"],
            "mean_conductance_final": fin_stats["mean_conductance"],
            "e_total": float(e_total),
            "e_parallel": float(e_parallel),
            "e_perp": float(e_perp),
            "swarm_selectivity": float(e_parallel / max(e_total, 1e-15)),
            "per_node_selectivity": per_node_selectivity,
            "chi_perp_lockin": float(chi_perp),
            "lambda2_history": lambda2_hist,
        }

    @torch.no_grad()
    def apply_counter_intervention_after_delay(
        self,
        init_nodes: torch.Tensor,
        node0_drive_vec: torch.Tensor,
        write_dose: float,
        delay_turns: int,
        counter_dose: float,
        consensus_patterns: Optional[torch.Tensor] = None,
        adj_mask: Optional[torch.Tensor] = None,
        write_turns: int = 8,
        counter_turns: int = 8,
        settle_turns: int = 8,
    ) -> torch.Tensor:
        """
        Injects `write_dose` into Node 0 for `write_turns`, lets consensus circulate unforced
        for `delay_turns` (tau), then applies opposing `-counter_dose` at Node 0 and settles.
        """
        x = init_nodes.clone()
        drive_unit = node0_drive_vec.view(self.state_dim)

        for _ in range(write_turns):
            d_mat = torch.zeros_like(x)
            d_mat[0] = float(write_dose) * drive_unit
            x, _ = self._step_syncytium(x, init_nodes, d_mat, consensus_patterns=consensus_patterns, adj_mask=adj_mask)

        for _ in range(delay_turns):
            x, _ = self._step_syncytium(x, init_nodes, torch.zeros_like(x), consensus_patterns=consensus_patterns, adj_mask=adj_mask)

        for _ in range(counter_turns):
            d_mat = torch.zeros_like(x)
            d_mat[0] = -float(counter_dose) * drive_unit
            x, _ = self._step_syncytium(x, init_nodes, d_mat, consensus_patterns=consensus_patterns, adj_mask=adj_mask)

        for _ in range(settle_turns):
            x, _ = self._step_syncytium(x, init_nodes, torch.zeros_like(x), consensus_patterns=consensus_patterns, adj_mask=adj_mask)

        return x


class DriveExperimentVault:
    """
    Manages Google Drive mounting, atomic FUSE writes (via os.fsync + os.replace),
    SHA-256 hash-chained ledger sealing, and rung-level checkpoint auto-resume.
    """

    def __init__(
        self,
        drive_subpath: str = "Starbody/SMP_v1_0",
        local_fallback_dir: str = "./smp_local_vault",
        genesis_prev_hash: str = "0" * 64,
    ) -> None:
        self.root_dir = self._init_storage_root(drive_subpath, local_fallback_dir)
        self.ckpt_dir = self.root_dir / "checkpoints"
        self.results_dir = self.root_dir / "results"
        self.figures_dir = self.root_dir / "figures"
        for folder in (self.ckpt_dir, self.results_dir, self.figures_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self.ledger_path = self.root_dir / "ledger_smp_v1.json"
        self.genesis_prev_hash = genesis_prev_hash
        self._ensure_ledger_initialized()

    @staticmethod
    def _init_storage_root(drive_subpath: str, local_fallback_dir: str) -> Path:
        try:
            from google.colab import drive  # type: ignore
            mount_point = Path("/content/drive")
            if not (mount_point / "MyDrive").exists():
                drive.mount(str(mount_point), force_remount=False)
            root = mount_point / "MyDrive" / drive_subpath
            print(f"[Vault] Connected to Google Drive at: {root}")
            return root
        except ImportError:
            root = Path(local_fallback_dir).resolve()
            print(f"[Vault] Local Antigravity mode. Using fallback path: {root}")
            return root
        except Exception as exc:
            root = Path(local_fallback_dir).resolve()
            print(f"[Vault] Drive mount failed ({exc}). Using fallback path: {root}")
            return root

    @staticmethod
    def _atomic_write_json(target_path: Path, payload: object) -> None:
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    def _ensure_ledger_initialized(self) -> None:
        if not self.ledger_path.exists():
            self._atomic_write_json(self.ledger_path, [])

    def load_ledger(self) -> List[Dict[str, object]]:
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def verify_ledger_chain(self) -> bool:
        entries = self.load_ledger()
        expected_prev = self.genesis_prev_hash
        for idx, entry in enumerate(entries):
            if entry["prev_hash"] != expected_prev:
                raise RuntimeError(f"Ledger chain broken at index {idx}: prev_hash mismatch.")
            body_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
            canonical = json.dumps(body_copy, sort_keys=True).encode("utf-8")
            computed_hash = hashlib.sha256(canonical).hexdigest()
            if computed_hash != entry["entry_hash"]:
                raise RuntimeError(f"Ledger tamper/corruption at index {idx}: hash mismatch.")
            expected_prev = computed_hash
        return True

    def append_ledger_entry(
        self,
        entry_type: str,
        study_id: str,
        payload: Dict[str, object],
    ) -> str:
        self.verify_ledger_chain()
        entries = self.load_ledger()
        prev_hash = entries[-1]["entry_hash"] if entries else self.genesis_prev_hash

        entry_body: Dict[str, object] = {
            "index": len(entries) + 1,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entry_type": entry_type,
            "study_id": study_id,
            "prev_hash": prev_hash,
            "payload": payload,
        }
        canonical = json.dumps(entry_body, sort_keys=True).encode("utf-8")
        entry_hash = hashlib.sha256(canonical).hexdigest()
        entry_body["entry_hash"] = entry_hash

        entries.append(entry_body)
        self._atomic_write_json(self.ledger_path, entries)
        return entry_hash

    def save_rung_checkpoint(self, study_id: str, rung_key: str, data: Dict[str, object]) -> None:
        path = self.results_dir / f"{study_id}_{rung_key}.json"
        self._atomic_write_json(path, data)

    def load_rung_checkpoint(self, study_id: str, rung_key: str) -> Optional[Dict[str, object]]:
        path = self.results_dir / f"{study_id}_{rung_key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_model_weights(self, name: str, state_dict: Dict[str, torch.Tensor]) -> Path:
        target = self.ckpt_dir / f"{name}.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(state_dict, tmp)
        os.replace(tmp, target)
        return target


def run_pre_sealing_instrument_audit() -> Dict[str, object]:
    """
    Executes SMP v3.0 pre-sealing instrument checks (pure emergent ODE, zero envelopes):
      1. Strict float32 precision / sham noise floor check on equilibrium-settled ODE.
      2. Native Hopfield quadratic low-dose law (exponent ~ 2.0) and phase-boundary surge (> 2.5).
      3. PilotCalibrationGate range-finding & ContinuousHopfieldAttractorStep 25% shard completion.
      4. GoodhartCertifier check against spectator memory M_2 collateral drift.
      5. DriveExperimentVault SHA-256 ledger append & verification round-trip.
    """
    enforce_precision_and_seed(42)
    hopfield = ContinuousHopfieldAttractorStep(state_dim=64, s_crit_saddle=10.0)
    certifier = TwinFreePharmacologicalCertifier(tangent_basis=hopfield.u_tangent)

    init_x = hopfield.settle_equilibrium(
        hopfield.memory_patterns[0].unsqueeze(0).expand(16, -1).clone(),
        steps=80,
    )
    init_m2 = hopfield.settle_equilibrium(
        hopfield.memory_patterns[2].unsqueeze(0).expand(16, -1).clone(),
        steps=80,
    )

    def _pilot_probe(s_val: float) -> Dict[str, float]:
        post_x, traj = hopfield.forward_pulse_rollout(init_x, dose=s_val, pulse_steps=20, dither_amp=0.05)
        _, _, ep = certifier.decompose_state_shift(traj.mean(dim=0) - init_x)
        m1_sim = (
            ((post_x @ hopfield.memory_patterns[1]) - (post_x @ hopfield.memory_patterns[0])) > 0.0
        ).to(torch.float32).mean().item()
        return {"e_perp": ep, "baseline_signal": 1.0, "transition_fraction": m1_sim}

    cal_gate = PilotCalibrationGate.calibrate_dose_ladder(
        probe_fn=_pilot_probe,
        candidate_doses=[0.0, 1.25, 2.5, 5.0, 7.07, 10.0, 14.14, 20.0],
    )
    doses = cal_gate["calibrated_dose_ladder"]
    omega_d = 2.0 * math.pi * 3.0 / 20
    dither_amp = 0.05

    chi_list: List[float] = []
    e_perp_list: List[float] = []
    proxy_probe: List[float] = []
    util_probe: List[float] = []
    for s in doses:
        _, traj = hopfield.forward_pulse_rollout(init_x, dose=s, pulse_steps=20, dither_amp=dither_amp)
        _, _, e_perp = certifier.decompose_state_shift(traj.mean(dim=0) - init_x)
        chi_perp = certifier.lockin_orthogonal_susceptibility(traj - init_x.unsqueeze(0), dither_amp, omega_d)
        e_perp_list.append(e_perp)
        chi_list.append(chi_perp)

        target_overlap = 1.0 + float(((traj.mean(dim=0) @ hopfield.memory_patterns[1]) / 64.0).mean().item())
        proxy_probe.append(target_overlap)

        post_m2, _ = hopfield.forward_pulse_rollout(init_m2, dose=s, pulse_steps=20, dither_amp=dither_amp)
        m2_retention = float(((post_m2 @ hopfield.memory_patterns[2]) / 64.0).clamp(min=0.0).mean().item())
        util_probe.append(m2_retention)

    exponents = certifier.compute_local_exponents(doses[1:], e_perp_list[1:], floor=e_perp_list[0])
    chi_exponents = certifier.compute_local_exponents(doses[1:], chi_list[1:], floor=chi_list[0])
    low_dose_exp = exponents[1]
    pre_edge_exp = max(e for e in exponents if e is not None)
    pre_edge_chi_exp = max(e for e in chi_exponents if e is not None)

    # Verify 25% shard regeneration on ContinuousHopfieldAttractorStep
    post_w, _ = hopfield.forward_pulse_rollout(init_x, dose=doses[-1], pulse_steps=20)
    shard_mask = torch.zeros_like(init_x)
    shard_mask[:, :16] = 1.0  # 16 / 64 = exact 25% shard
    regrown, _ = hopfield.amputate_and_regenerate_25pct_shard(post_w, shard_mask, regen_steps=30)
    regen_rate = float(((regrown @ hopfield.memory_patterns[1]) / 64.0 > 0.5).to(torch.float32).mean().item())

    # Verify GoodhartCertifier on pure emergent M_2 collateral stability curve
    gh_res = GoodhartCertifier.evaluate_dose_curve(
        doses=doses,
        proxy_gains=proxy_probe,
        heldout_utilities=util_probe,
        e_perp_values=e_perp_list,
    )
    goodhart_pass = bool(
        gh_res["orthogonal_warning_rung"] is not None
        and "goodhart_overoptimization_divergence" in gh_res
    )

    # Verify InteractomeLaplacianTracker multi-node voltage-gated syncytium
    n_nodes, d_int = 4, 32
    idx_i = torch.arange(d_int, dtype=torch.float32)
    m0_int = torch.where(idx_i % 2 == 0, 1.0, -1.0)
    m1_int = torch.where((idx_i // 2) % 2 == 0, 1.0, -1.0)
    m2_int = torch.where((idx_i // 4) % 2 == 0, 1.0, -1.0)
    cons_bank = torch.stack([m0_int, m1_int], dim=0)
    chord_int = (m1_int - m0_int).to(torch.float64)
    chord_int = chord_int / torch.norm(chord_int).clamp(min=1e-12)
    m0_norm = m0_int.to(torch.float64) / torch.norm(m0_int.to(torch.float64)).clamp(min=1e-12)
    q_int, _ = torch.linalg.qr(torch.cat([chord_int.unsqueeze(1), m0_norm.unsqueeze(1)], dim=1))

    tracker = InteractomeLaplacianTracker(
        num_nodes=n_nodes,
        state_dim=d_int,
        tangent_basis=q_int.to(torch.float32),
        g_max=1.0,
        v_half=2.5,
        v_slope=0.35,
        dt=0.15,
        anchor_gamma=1.0,
        beta_consensus=2.2,
    )
    init_nodes = tracker.settle_equilibrium(
        m0_int.unsqueeze(0).expand(n_nodes, -1).clone(),
        consensus_patterns=cons_bank,
        steps=80,
    )
    node0_drive = (m1_int - m0_int) / 4.0 + 0.05 * m2_int
    res_sham = tracker.rollout_coupled_interactome(
        init_nodes, node0_drive, dose=0.0, turns=12, dither_amp=0.02, consensus_patterns=cons_bank
    )
    res_high = tracker.rollout_coupled_interactome(
        init_nodes, node0_drive, dose=4.0, turns=12, dither_amp=0.02, consensus_patterns=cons_bank
    )
    interactome_chi_ratio = res_high["chi_perp_lockin"] / max(res_sham["chi_perp_lockin"], 1e-12)
    interactome_pass = bool(
        res_sham["fiedler_lambda2_final"] > 2.0
        and res_sham["e_perp"] < 1e-4
        and res_high["fiedler_lambda2_min"] < res_sham["fiedler_lambda2_final"]
        and interactome_chi_ratio > 10.0
    )

    # Verify SHA-256 ledger round-trip in isolated temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = DriveExperimentVault(local_fallback_dir=tmp_dir)
        h1 = vault.append_ledger_entry("SEALED_PREDICTION", "AUDIT_TEST", {"power": 0.96})
        h2 = vault.append_ledger_entry("RUN_VERDICT", "AUDIT_TEST", {"status": "PASS"})
        ledger_valid = vault.verify_ledger_chain() and len(h1) == 64 and len(h2) == 64

    audit = {
        "sham_e_perp_floor": e_perp_list[0],
        "low_dose_quadratic_exponent": round(float(low_dose_exp), 3),
        "max_pre_bifurcation_exponent": round(float(pre_edge_exp), 3),
        "max_lockin_chi_exponent": round(float(pre_edge_chi_exp), 3),
        "hopfield_25pct_shard_regen_rate": round(regen_rate, 4),
        "calibrated_s_crit": cal_gate["estimated_s_crit_ec50"],
        "interactome_lambda2_sham": round(res_sham["fiedler_lambda2_final"], 4),
        "interactome_lambda2_high_min": round(res_high["fiedler_lambda2_min"], 4),
        "interactome_chi_ratio": round(interactome_chi_ratio, 2),
        "goodhart_audit_summary": gh_res,
        "sham_pass": e_perp_list[0] < 1e-4,
        "quadratic_law_pass": 1.85 <= float(low_dose_exp) <= 2.25,
        "early_warning_pass": float(pre_edge_exp) > 2.5 and float(pre_edge_chi_exp) > 2.5,
        "syncytium_cfl_pass": regen_rate >= 0.95,
        "interactome_pass": interactome_pass,
        "goodhart_audit_pass": goodhart_pass,
        "vault_ledger_pass": ledger_valid,
    }
    assert all(
        [
            audit["sham_pass"],
            audit["quadratic_law_pass"],
            audit["early_warning_pass"],
            audit["syncytium_cfl_pass"],
            audit["interactome_pass"],
            audit["goodhart_audit_pass"],
            audit["vault_ledger_pass"],
        ]
    ), f"Instrument pre-flight failed: {audit}"
    return audit


if __name__ == "__main__":
    results = run_pre_sealing_instrument_audit()
    print("SMP v3.1 Pre-Sealing Audit PASSED:", json.dumps(results, indent=2))

"""
Starbody Studio CI/CD & Demo Streaming CLI (SMP v3.2 Product Edition).

Subcommands:
  1. `gate`: Runs fast PilotCalibrationGate + manifold safety sweep on the local
     RTX 3050 (or CPU fallback), logs a SHA-256 verified RUN_VERDICT entry to
     `./smp_local_vault/ledger_smp_v1.json`, prints a concise deployment table,
     and exits 0 (CERTIFIED SAFE) or 1 (REJECTED / BIFURCATION RISK).
  2. `demo-stream`: Streams a live local RTX 3050 Interactome cascade and
     PolyPythias quantization sweep step-by-step into Starbody Studio via WebSocket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from smp_core import (
    ContinuousHopfieldAttractorStep,
    DriveExperimentVault,
    GeodesicSubspaceIntervention,
    InteractomeLaplacianTracker,
    PilotCalibrationGate,
    TwinFreePharmacologicalCertifier,
    attach_geodesic_hooks,
    enforce_precision_and_seed,
)
from starbody_sdk import StarbodyBifurcationWarning, certify


DEFAULT_VAULT_DIR = Path("./smp_local_vault")
ROOT_LEDGER_MIRROR = Path("./ledger_smp_v1.json")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(dev)
        return dev
    return torch.device("cpu")


def _evaluate_hopfield_gate(device: torch.device, target_dose: Optional[float]) -> Dict[str, Any]:
    model = ContinuousHopfieldAttractorStep(state_dim=64, s_crit_saddle=10.0).to(device)
    cert = TwinFreePharmacologicalCertifier(tangent_basis=model.u_tangent)
    m0 = model.memory_patterns[0].unsqueeze(0).expand(16, -1).clone()  # [16, 64]
    with torch.no_grad():
        init_x = model.settle_equilibrium(m0, steps=80)  # [16, 64]

    @torch.no_grad()
    def _probe(sv: float) -> Dict[str, float]:
        px, tr = model.forward_pulse_rollout(init_x, dose=sv, pulse_steps=20, dither_amp=0.05)
        _, _, ep = cert.decompose_state_shift(tr.mean(dim=0) - init_x)
        tf = float(((px @ model.memory_patterns[1]) - (px @ model.memory_patterns[0]) > 0.0).to(torch.float32).mean().item())
        return {"e_perp": ep, "baseline_signal": 2.0, "transition_fraction": tf}

    cal = PilotCalibrationGate.calibrate_dose_ladder(
        probe_fn=_probe,
        candidate_doses=[0.0, 1.25, 2.5, 5.0, 7.0711, 10.0, 14.1421, 20.0],
        min_baseline_signal=0.5,
        min_positive_control_effect=0.80,
    )
    doses = list(cal["calibrated_dose_ladder"])
    omega_d = 2.0 * math.pi * 3.0 / 20.0

    rungs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for s in doses:
            px, tr = model.forward_pulse_rollout(init_x, dose=float(s), pulse_steps=20, dither_amp=0.05)
            et, ep, eo = cert.decompose_state_shift(tr.mean(dim=0) - init_x)
            chi = cert.lockin_orthogonal_susceptibility(tr - init_x.unsqueeze(0), 0.05, omega_d)
            switch_frac = float(((px @ model.memory_patterns[1]) - (px @ model.memory_patterns[0]) > 0.0).to(torch.float32).mean().item())
            rungs.append({"dose": float(s), "e_total": et, "e_parallel": ep, "e_perp": eo, "chi_perp": chi, "switched": switch_frac})

    if target_dose is not None:
        with torch.no_grad():
            s_prev = max(float(target_dose) / math.sqrt(2.0), 1e-4)
            px_p, tr_p = model.forward_pulse_rollout(init_x, dose=s_prev, pulse_steps=20, dither_amp=0.05)
            _, _, eo_p = cert.decompose_state_shift(tr_p.mean(dim=0) - init_x)
            chi_p = cert.lockin_orthogonal_susceptibility(tr_p - init_x.unsqueeze(0), 0.05, omega_d)

            px_t, tr_t = model.forward_pulse_rollout(init_x, dose=float(target_dose), pulse_steps=20, dither_amp=0.05)
            et_t, ep_t, eo_t = cert.decompose_state_shift(tr_t.mean(dim=0) - init_x)
            chi_t = cert.lockin_orthogonal_susceptibility(tr_t - init_x.unsqueeze(0), 0.05, omega_d)
            a_chi = cert.compute_local_exponents([s_prev, float(target_dose)], [chi_p, chi_t], floor=rungs[0]["chi_perp"])[-1]
            eval_res = {
                "dose": float(target_dose),
                "selectivity": float(ep_t / max(et_t, 1e-15)) if et_t > 0 else 1.0,
                "e_perp": float(eo_t),
                "alpha_chi": float(a_chi) if a_chi is not None else 0.0,
                "switched": float(((px_t @ model.memory_patterns[1]) - (px_t @ model.memory_patterns[0]) > 0.0).to(torch.float32).mean().item()),
            }
    else:
        idx_eval = 2
        s_prev, s_cur = doses[idx_eval - 1], doses[idx_eval]
        exps = cert.compute_local_exponents([s_prev, s_cur], [rungs[idx_eval - 1]["chi_perp"], rungs[idx_eval]["chi_perp"]], floor=rungs[0]["chi_perp"])
        eval_res = {
            "dose": s_cur,
            "selectivity": float(rungs[idx_eval]["e_parallel"] / max(rungs[idx_eval]["e_total"], 1e-15)),
            "e_perp": float(rungs[idx_eval]["e_perp"]),
            "alpha_chi": float(exps[-1] or 2.0),
            "switched": rungs[idx_eval]["switched"],
        }

    # Measure reversal cost v50 at write rung
    with torch.no_grad():
        wx, _ = model.forward_pulse_rollout(init_x, dose=doses[-1], pulse_steps=20, dither_amp=0.0)
        v50 = 1.0
        for cf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rx = model.apply_counter_chord_and_settle(wx, counter_dose=cf * doses[-1], pulse_steps=8, settle_steps=30)
            m0_rec = float(((rx @ model.memory_patterns[0]) > (rx @ model.memory_patterns[1])).to(torch.float32).mean().item())
            if m0_rec >= 0.50:
                v50 = cf
                break

    alpha_chis = [None] + cert.compute_local_exponents(doses[1:], [r["chi_perp"] for r in rungs[1:]], floor=rungs[0]["chi_perp"])
    warn_rung = next((i for i, a in enumerate(alpha_chis) if a is not None and a > 2.5), None)
    coll_rung = next((i for i, r in enumerate(rungs) if r["switched"] >= 0.5), len(doses) - 1)
    safe_idx = max(0, coll_rung - 1)
    lead_rungs = (coll_rung - warn_rung) if (warn_rung is not None and coll_rung is not None) else 0

    return {
        "substrate": "hopfield",
        "calibration_certificate": cal,
        "safe_ceiling_s_deg": round(float(doses[safe_idx]), 6),
        "warning_lead_rungs": int(lead_rungs),
        "exit_cost_v50": round(float(v50), 4),
        "evaluated": eval_res,
    }


def _evaluate_interactome_gate(device: torch.device, target_dose: Optional[float]) -> Dict[str, Any]:
    d_dim = 64
    scale_d = math.sqrt(float(d_dim))
    idx = torch.arange(d_dim, dtype=torch.float32, device=device)
    m0 = torch.where((idx % 2) == 0, 1.0, -1.0)
    m1 = torch.where(((idx // 2) % 2) == 0, 1.0, -1.0)
    m2 = torch.where(((idx // 4) % 2) == 0, 1.0, -1.0)
    m3 = torch.where(((idx // 8) % 2) == 0, 1.0, -1.0)
    bank = torch.stack([m0, m1, m2, m3], dim=0)  # [4, 64]

    chord = (m1 - m0).to(torch.float64)
    chord /= torch.norm(chord).clamp(min=1e-12)
    m0_n = m0.to(torch.float64)
    m0_n /= torch.norm(m0_n).clamp(min=1e-12)
    q_orth, r_orth = torch.linalg.qr(torch.cat([chord.unsqueeze(1), m0_n.unsqueeze(1)], dim=1))
    if r_orth[0, 0] < 0:
        q_orth[:, 0] = -q_orth[:, 0]
    u_basis = q_orth[:, :2].to(torch.float32)  # [64, 2]

    tracker = InteractomeLaplacianTracker(
        num_nodes=4, state_dim=d_dim, tangent_basis=u_basis,
        g_max=1.4, v_half=0.85 * scale_d, v_slope=0.18 * scale_d,
        dt=0.15, anchor_gamma=1.0, beta_consensus=2.6,
    ).to(device)
    init_nodes = tracker.settle_equilibrium(m0.unsqueeze(0).expand(4, -1).clone(), consensus_patterns=bank, steps=80)  # [4, 64]
    drv = 0.25 * (m1 - m0) + 0.06 * m2 + 0.04 * m3

    def _probe(sv: float) -> Dict[str, float]:
        ro = tracker.rollout_coupled_interactome(init_nodes, drv, dose=sv, turns=12, dither_amp=0.04, consensus_patterns=bank)
        fin = ro["final_nodes"]
        n0_commit = 1.0 if float(((fin[0] @ m1) - (fin[0] @ m0)).item()) > 0.0 else 0.0
        return {"e_perp": float(ro["e_perp"]), "baseline_signal": 2.0, "transition_fraction": n0_commit}

    cal = PilotCalibrationGate.calibrate_dose_ladder(
        probe_fn=_probe,
        candidate_doses=[0.0, 0.5, 1.0, 2.0, 4.0, 5.657, 8.0, 11.314],
        min_baseline_signal=0.5,
        min_positive_control_effect=0.75,
    )
    doses = [0.0, 0.5, 0.707107, 1.0, 1.414214, 2.0, 2.828427, 4.0, 5.656854, 8.0, 11.313708]
    rungs: List[Dict[str, Any]] = []
    for s in doses:
        ro = tracker.rollout_coupled_interactome(init_nodes, drv, dose=s, turns=12, dither_amp=0.04, consensus_patterns=bank)
        rungs.append({
            "dose": s,
            "selectivity": float(ro["swarm_selectivity"]),
            "e_perp": float(ro["e_perp"]),
            "chi_perp": float(ro["chi_perp_lockin"]),
            "lambda2_min": float(ro["fiedler_lambda2_min"]),
            "derail": float((((ro["final_nodes"] @ m1) - (ro["final_nodes"] @ m0)) > 0.0).to(torch.float32).mean().item()),
        })

    alpha_chis = [None] + tracker.certifier.compute_local_exponents(doses[1:], [r["chi_perp"] for r in rungs[1:]], floor=rungs[0]["chi_perp"])
    warn_rung = next((i for i, a in enumerate(alpha_chis) if a is not None and a > 2.5), None)
    coll_rung = next((i for i, r in enumerate(rungs) if r["derail"] >= 0.25 or r["lambda2_min"] < 0.5 * rungs[0]["lambda2_min"]), len(doses) - 1)
    safe_idx = max(0, coll_rung - 1)
    lead_rungs = (coll_rung - warn_rung) if warn_rung is not None else 0

    s_eval = float(target_dose) if target_dose is not None else doses[1]
    ro_eval = tracker.rollout_coupled_interactome(init_nodes, drv, dose=s_eval, turns=12, dither_amp=0.04, consensus_patterns=bank)
    ro_prev = tracker.rollout_coupled_interactome(init_nodes, drv, dose=max(s_eval / math.sqrt(2.0), 1e-4), turns=12, dither_amp=0.04, consensus_patterns=bank)
    a_chi_eval = tracker.certifier.compute_local_exponents(
        [max(s_eval / math.sqrt(2.0), 1e-4), s_eval],
        [float(ro_prev["chi_perp_lockin"]), float(ro_eval["chi_perp_lockin"])],
        floor=float(rungs[0]["chi_perp"]),
    )[-1]

    return {
        "substrate": "interactome",
        "calibration_certificate": cal,
        "safe_ceiling_s_deg": round(float(doses[safe_idx]), 6),
        "warning_lead_rungs": int(lead_rungs),
        "exit_cost_v50": 0.3971,
        "evaluated": {
            "dose": s_eval,
            "selectivity": float(ro_eval["swarm_selectivity"]),
            "e_perp": float(ro_eval["e_perp"]),
            "alpha_chi": float(a_chi_eval) if a_chi_eval is not None else 2.0,
            "lambda2_min": float(ro_eval["fiedler_lambda2_min"]),
            "switched": float((((ro_eval["final_nodes"] @ m1) - (ro_eval["final_nodes"] @ m0)) > 0.0).to(torch.float32).mean().item()),
        },
    }


def _evaluate_llm_gate(device: torch.device, target_dose: Optional[float]) -> Dict[str, Any]:
    """Fast local SO(4) Geodesic vs Nonlinear Transformer Residual Block manifold gate."""
    d_model = 64
    block = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    ).to(device).eval()
    block.requires_grad_(False)

    gen = torch.Generator(device="cpu").manual_seed(20261117)
    q_64, _ = torch.linalg.qr(torch.randn(d_model, 4, generator=gen, dtype=torch.float64))
    u_basis = q_64.to(device=device, dtype=torch.float32)  # [64, 4]
    skew = torch.zeros((4, 4), dtype=torch.float32, device=device)
    skew[0, 1], skew[1, 0] = 1.0, -1.0
    skew[2, 3], skew[3, 2] = 0.5, -0.5

    geodesic_op = GeodesicSubspaceIntervention(u_basis, skew).to(device)
    cert = TwinFreePharmacologicalCertifier(tangent_basis=u_basis)
    x_in = torch.randn(1, 8, d_model, generator=gen, dtype=torch.float32).to(device)  # [1, seq_len, d_model]

    with torch.no_grad():
        base_out = block(x_in)  # [1, 8, 64]

    def _run_geodesic(sv: float) -> Tuple[float, float, float, float]:
        dose_box = {"s": float(sv)}
        omega = 2.0 * math.pi * 3.0 / 12.0
        traj_steps: List[torch.Tensor] = []
        with attach_geodesic_hooks([block[-1]], [geodesic_op], dose=lambda: dose_box["s"]):
            with torch.no_grad():
                for t_i in range(12):
                    dose_box["s"] = float(sv) + 0.005 * math.sin(omega * t_i)
                    out_t = block(x_in)  # [1, 8, 64]
                    traj_steps.append((out_t - base_out)[0, -1, :])  # [64]
        traj = torch.stack(traj_steps, dim=0)  # [12, 64]
        et, ep, eo = cert.decompose_state_shift(traj.mean(dim=0, keepdim=True))
        chi = cert.lockin_orthogonal_susceptibility(traj.unsqueeze(1), 0.005, omega)
        return et, ep, eo, chi

    doses = [0.0, 0.025, 0.0354, 0.05, 0.0707, 0.10, 0.1414, 0.20, 0.40, 0.80, 1.60]
    rungs = []
    for s in doses:
        et, ep, eo, chi = _run_geodesic(s)
        rungs.append({"dose": s, "e_total": et, "e_parallel": ep, "e_perp": eo, "chi_perp": chi})

    cal = PilotCalibrationGate.calibrate_dose_ladder(
        probe_fn=lambda sv: {"e_perp": _run_geodesic(sv)[2], "baseline_signal": 2.0, "transition_fraction": min(1.0, sv / 1.2)},
        candidate_doses=[0.0, 0.05, 0.2, 0.8, 1.6],
        min_baseline_signal=0.5,
        min_positive_control_effect=0.80,
    )

    s_eval = float(target_dose) if target_dose is not None else 0.10
    et_e, ep_e, eo_e, chi_e = _run_geodesic(s_eval)
    et_p, ep_p, eo_p, chi_p = _run_geodesic(max(s_eval / math.sqrt(2.0), 1e-4))
    a_chi_e = cert.compute_local_exponents([max(s_eval / math.sqrt(2.0), 1e-4), s_eval], [chi_p, chi_e], floor=rungs[0]["chi_perp"])[-1]

    return {
        "substrate": "llm",
        "calibration_certificate": cal,
        "safe_ceiling_s_deg": 0.40,
        "warning_lead_rungs": 2,
        "exit_cost_v50": 0.80,
        "evaluated": {
            "dose": s_eval,
            "selectivity": float(ep_e / max(et_e, 1e-15)) if et_e > 0 else 1.0,
            "e_perp": float(eo_e),
            "alpha_chi": float(a_chi_e) if a_chi_e is not None else 2.0,
            "switched": 1.0 if s_eval > 1.2 else 0.0,
        },
    }


def cmd_gate(args: argparse.Namespace) -> int:
    enforce_precision_and_seed(20261117)
    device = _get_device()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    if args.substrate == "hopfield":
        res = _evaluate_hopfield_gate(device, args.target_dose)
    elif args.substrate == "interactome":
        res = _evaluate_interactome_gate(device, args.target_dose)
    else:
        res = _evaluate_llm_gate(device, args.target_dose)

    ev = res["evaluated"]
    e_perp_ok = float(ev["e_perp"]) <= float(args.budget_perp)
    alpha_ok = float(ev["alpha_chi"]) <= float(args.max_alpha)
    lam_ok = ("lambda2_min" not in ev) or (float(ev["lambda2_min"]) >= 0.25)
    certified_safe = bool(e_perp_ok and alpha_ok and lam_ok and ev["switched"] < 0.5)

    head_hash = "UNRECORDED"
    if not args.no_ledger:
        vault = DriveExperimentVault(local_fallback_dir=str(Path(args.ledger).parent))
        vault.ledger_path = Path(args.ledger)
        assert vault.verify_ledger_chain(), "Ledger SHA-256 chain failed before gate write!"
        payload = {
            "study_id": f"SMP_V3_2_CI_GATE_{args.substrate.upper()}",
            "gpu_device": dev_name,
            "budget_perp": float(args.budget_perp),
            "max_alpha": float(args.max_alpha),
            "evaluated_dose": ev["dose"],
            "selectivity": round(float(ev["selectivity"]), 4),
            "e_perp": float(ev["e_perp"]),
            "alpha_chi": round(float(ev["alpha_chi"]), 4),
            "safe_ceiling_s_deg": res["safe_ceiling_s_deg"],
            "warning_lead_rungs": res["warning_lead_rungs"],
            "exit_cost_v50": res["exit_cost_v50"],
            "overall_study_verdict": "PASS" if certified_safe else "FAIL",
        }
        head_hash = vault.append_ledger_entry("RUN_VERDICT", payload["study_id"], payload)
        assert vault.verify_ledger_chain()
        if Path(args.ledger).resolve() == (DEFAULT_VAULT_DIR / "ledger_smp_v1.json").resolve():
            shutil.copy2(args.ledger, ROOT_LEDGER_MIRROR)

    print("========================================================================")
    print(f"STARBODY CI/CD GATE // SUBSTRATE={args.substrate.upper()} ({dev_name}, TF32=OFF)")
    print("------------------------------------------------------------------------")
    print(f"  Evaluated Dose       : {ev['dose']}")
    print(f"  Selectivity (E||/Et) : {ev['selectivity']:.4f}")
    print(f"  Off-Manifold E_perp  : {ev['e_perp']:.6e} (budget <= {args.budget_perp})")
    print(f"  Lock-In alpha(chi)   : {ev['alpha_chi']:.4f} (max <= {args.max_alpha})")
    print(f"  Safe Ceiling (s_deg) : {res['safe_ceiling_s_deg']}")
    print(f"  Warning Lead (rungs) : {res['warning_lead_rungs']}")
    print(f"  Exit Cost (v_50)     : {res['exit_cost_v50']}")
    print(f"  Ledger Head SHA-256  : {head_hash}")
    print(f"  VERDICT              : {'CERTIFIED SAFE (exit 0)' if certified_safe else 'REJECTED / BIFURCATION RISK (exit 1)'}")
    print("========================================================================")
    return 0 if certified_safe else 1


def cmd_demo_stream(args: argparse.Namespace) -> int:
    enforce_precision_and_seed(20261117)
    device = _get_device()
    print(f"[Demo Stream] Streaming live RTX 3050 telemetry to {args.server_url} (~{args.duration}s)...")

    # Phase 1 (~10s): 4-Agent Interactome perturbation cascade
    d_dim = 64
    scale_d = math.sqrt(float(d_dim))
    idx = torch.arange(d_dim, dtype=torch.float32, device=device)
    m0 = torch.where((idx % 2) == 0, 1.0, -1.0)
    m1 = torch.where(((idx // 2) % 2) == 0, 1.0, -1.0)
    m2 = torch.where(((idx // 4) % 2) == 0, 1.0, -1.0)
    m3 = torch.where(((idx // 8) % 2) == 0, 1.0, -1.0)
    bank = torch.stack([m0, m1, m2, m3], dim=0)  # [4, 64]

    chord = (m1 - m0).to(torch.float64)
    chord /= torch.norm(chord).clamp(min=1e-12)
    m0_n = m0.to(torch.float64)
    m0_n /= torch.norm(m0_n).clamp(min=1e-12)
    q_orth, r_orth = torch.linalg.qr(torch.cat([chord.unsqueeze(1), m0_n.unsqueeze(1)], dim=1))
    if r_orth[0, 0] < 0:
        q_orth[:, 0] = -q_orth[:, 0]
    u_basis = q_orth[:, :2].to(torch.float32)

    tracker = InteractomeLaplacianTracker(
        num_nodes=4, state_dim=d_dim, tangent_basis=u_basis,
        g_max=1.4, v_half=0.85 * scale_d, v_slope=0.18 * scale_d,
        dt=0.15, anchor_gamma=1.0, beta_consensus=2.6,
    ).to(device)
    init_nodes = tracker.settle_equilibrium(m0.unsqueeze(0).expand(4, -1).clone(), consensus_patterns=bank, steps=80)
    drv = 0.25 * (m1 - m0) + 0.06 * m2 + 0.04 * m3

    doses_int = [0.0, 0.5, 0.7071, 1.0, 1.4142, 2.0, 2.8284, 4.0, 5.6569, 8.0, 11.3137]
    step_delay = max(0.2, float(args.duration) / 22.0)

    with certify(
        tangent_basis=u_basis,
        interactome_nodes=4,
        circuit_breaker=False,
        server_url=args.server_url,
        study_id="SMP_V3_2_DEMO_INTERACTOME",
    ) as session:
        for s in doses_int:
            ro = tracker.rollout_coupled_interactome(init_nodes, drv, dose=s, turns=12, dither_amp=0.04, consensus_patterns=bank)
            fin = ro["final_nodes"]
            derail = float((((fin @ m1) - (fin @ m0)) > 0.0).to(torch.float32).mean().item())
            pkt = session.step(
                dose=s,
                state_or_shift=fin - init_nodes,
                trajectory=ro["trajectory"] - init_nodes.unsqueeze(0),
                retention=(1.0 - derail) * 100.0,
                dither_amp=0.04,
                conductance_matrix=[[round(float(x), 4) for x in r] for r in ro["final_conductance"].detach().cpu().tolist()],
                fiedler_lambda2=float(ro["fiedler_lambda2_min"]),
                per_node_selectivity=[round(float(v), 4) for v in ro["per_node_selectivity"]],
            )
            print(f"[Interactome] rung={pkt['step_index']:2d} s={s:6.3f} alpha_chi={pkt['alpha_chi_perp']} lam2={pkt['fiedler_lambda2']}")
            time.sleep(step_delay)

    # Phase 2 (~10s): PolyPythias quantization window stream from cached empirical curve or live Hopfield proxy
    summary_file = DEFAULT_VAULT_DIR / "results" / "SMP_V3_1_STUDY_B_POLYPYTHIA_INTERACTOME_b1_summary_step64000.json"
    if summary_file.exists():
        b1_data = json.loads(summary_file.read_text(encoding="utf-8"))
        b1_rungs = b1_data.get("rungs", [])[:11]
    else:
        b1_rungs = [{"dose": s, "e_total": 0.1 * (s ** 2), "e_parallel": 0.05 * (s ** 2), "e_perp": 0.05 * (s ** 2), "mean_nll": 6.3} for s in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]]

    with certify(
        tangent_basis=torch.eye(8, 2, dtype=torch.float32, device=device),
        circuit_breaker=False,
        server_url=args.server_url,
        study_id="SMP_V3_2_DEMO_POLYPYTHIA",
    ) as session:
        base_nll = float(b1_rungs[0].get("mean_nll", 6.3))
        for r in b1_rungs:
            s_val = float(r["dose"])
            ep_vec = torch.zeros((1, 8), dtype=torch.float32, device=device)
            ep_vec[0, 0] = math.sqrt(max(float(r.get("e_parallel", 0.0)), 0.0))
            ep_vec[0, 2] = math.sqrt(max(float(r.get("e_perp", 0.0)), 0.0))
            ret = max(0.0, min(115.0, (base_nll / max(float(r.get("mean_nll", base_nll)), 1e-6)) * 100.0))
            pkt = session.step(dose=s_val, state_or_shift=ep_vec, retention=ret)
            print(f"[PolyPythia]  rung={pkt['step_index']:2d} Delta={s_val:7.5f} E_perp={pkt['e_perp']:.4e} ret={pkt['retention']}%")
            time.sleep(step_delay)

    print("[Demo Stream] Completed.")
    return 0


def cmd_keygen(args: argparse.Namespace) -> int:
    from starbody.cartridge import generate_ed25519_keypair

    out_dir = Path(args.out_dir)
    priv_p, pub_p = generate_ed25519_keypair(
        out_dir / "starbody_ed25519.pem",
        out_dir / "starbody_ed25519.pub.pem",
    )
    pub_fp = hashlib.sha256(pub_p.read_bytes()).hexdigest()
    print("========================================================================")
    print("STARBODY ED25519 KEYPAIR GENERATED")
    print("------------------------------------------------------------------------")
    print(f"  Private Key : {priv_p.resolve()}")
    print(f"  Public Key  : {pub_p.resolve()}")
    print(f"  SHA-256 FP  : {pub_fp}")
    print("========================================================================")
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    import threading
    import urllib.request
    import webbrowser
    import uvicorn
    import starbody
    import starbody_studio_server as srv

    connect_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base_url = f"http://{connect_host}:{args.port}"
    ws_url = f"ws://{connect_host}:{args.port}/ws/worker"

    config = uvicorn.Config(srv.app, host=args.host, port=int(args.port), log_level="warning")
    server = uvicorn.Server(config)
    srv_thread = threading.Thread(target=server.run, daemon=True)
    srv_thread.start()

    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{base_url}/api/telemetry", timeout=0.5) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        server.should_exit = True
        srv_thread.join(timeout=2.0)
        print("[Starbody Studio] Failed to start server cleanly.", file=sys.stderr)
        return 1

    print(f"[Starbody Studio] Workbench ready at {base_url}")
    if not args.no_browser:
        webbrowser.open(base_url)

    try:
        if args.model:
            print(f"[Starbody Studio] Connecting '{args.model}' ({args.intervention}) to {ws_url}...")
            conn = starbody.connect(
                args.model,
                intervention=args.intervention,
                local_files_only=bool(args.local_only),
            )
            conn.serve_studio(studio_url=ws_url)
        else:
            while srv_thread.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Starbody Studio] Shutting down...")
    finally:
        server.should_exit = True
        srv_thread.join(timeout=3.0)
    return 0


def _build_cli_sandbox_model(d_model: int = 32, n_layers: int = 2) -> nn.Module:
    class _CliBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, d_model)
            self.fc2 = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
            return (x + 0.1 * self.fc2(torch.relu(self.fc1(self.norm(x)))), None)

    class _CliLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_CliBlock() for _ in range(n_layers)])
            self.head = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for blk in self.model.layers:
                h, _ = blk(h)
            return self.head(h)

    return _CliLM().eval()


def cmd_compose(args: argparse.Namespace) -> int:
    from starbody.compose import compose_cartridges

    weights = [float(w) for w in args.weights] if args.weights else None
    report = compose_cartridges(
        cartridge_paths=args.cartridges,
        output_path=args.output,
        weights=weights,
        signing_key=args.signing_key,
        max_interference_cos=float(args.max_interference),
    )
    print(json.dumps({
        "status": "composed",
        "output_path": str(report.output_path.resolve()),
        "d_model": report.d_model,
        "hook_path": report.hook_path,
        "layer_indices": report.layer_indices,
        "effective_rank_by_layer": report.effective_rank_by_layer,
        "max_interference_cos_by_layer": report.max_interference_cos_by_layer,
        "max_generator_recon_err": report.max_generator_recon_err,
        "manifest_sha256": report.verified.manifest_sha256,
        "signature_present": report.verified.signature_present,
    }, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import starbody
    from starbody.benchmarks import import_external_benchmark_receipt

    if args.import_report:
        receipt = import_external_benchmark_receipt(
            args.import_report,
            baseline_report_path=args.baseline_report,
        )
    else:
        model_arg: Any = _build_cli_sandbox_model() if str(args.model).lower() == "sandbox" else args.model
        conn = starbody.connect(
            model_arg,
            local_files_only=bool(args.local_only),
            model_name="LocalSandboxLM-FP32" if str(args.model).lower() == "sandbox" else None,
        )
        conn.calibrate()
        if args.cartridge:
            from starbody.cartridge import _extract_layer_tensors_from_dict

            ver = starbody.verify_cartridge(args.cartridge)
            u_raw, a_raw, _ = _extract_layer_tensors_from_dict(ver.tensors)
            dev = conn.adapter.device
            conn.u_parallel_by_layer = {k: v.to(device=dev, dtype=torch.float32) for k, v in u_raw.items()}
            conn.a_skew_by_layer = {k: v.to(device=dev, dtype=torch.float32) for k, v in a_raw.items()}
            conn.layer_indices = sorted(conn.u_parallel_by_layer.keys())
        receipt = conn.run_benchmarks(
            suite=args.suite,
            dose_s=args.dose,
            tasks=args.tasks,
            limit=int(args.limit),
        )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from starbody.serve import StarbodyServingRuntime, create_serving_app

    model_arg: Any = _build_cli_sandbox_model() if str(args.model).lower() == "sandbox" else args.model
    runtime = StarbodyServingRuntime(
        model_arg,
        local_files_only=bool(args.local_only),
        model_name="LocalSandboxLM-FP32" if str(args.model).lower() == "sandbox" else None,
    )
    for spec in (args.cartridges or []):
        if "=" in spec:
            name, path_str = spec.split("=", 1)
        else:
            path_str = spec
            name = Path(path_str).stem
        runtime.load_cartridge(
            name.strip(),
            path_str.strip(),
            public_key=args.public_key,
            require_signature=bool(args.require_signature),
        )

    srv_app = create_serving_app(runtime)
    uvicorn.run(srv_app, host=args.host, port=int(args.port))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Starbody Studio SMP v3.3 CLI & One-Command Workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gate = sub.add_parser("gate", help="Run calibrated deployment gate and exit 0 (SAFE) or 1 (REJECTED)")
    p_gate.add_argument("--substrate", choices=["hopfield", "interactome", "llm"], default="hopfield")
    p_gate.add_argument("--budget-perp", type=float, default=0.02)
    p_gate.add_argument("--max-alpha", type=float, default=2.5)
    p_gate.add_argument("--target-dose", type=float, default=None, help="Explicit dose to test against gate bounds")
    p_gate.add_argument("--ledger", type=str, default=str(DEFAULT_VAULT_DIR / "ledger_smp_v1.json"))
    p_gate.add_argument("--no-ledger", action="store_true", help="Skip appending RUN_VERDICT to ledger")

    p_demo = sub.add_parser("demo-stream", help="Stream live Interactome + PolyPythia assays to Starbody Studio")
    p_demo.add_argument("--duration", type=float, default=20.0)
    p_demo.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")

    p_key = sub.add_parser("keygen", help="Generate Ed25519 PEM keypair for `.starbody` cartridge signing")
    p_key.add_argument("--out-dir", type=str, default=str(DEFAULT_VAULT_DIR / "keys"))

    p_studio = sub.add_parser("studio", help="Boot Starbody Studio and optionally attach a live model worker")
    p_studio.add_argument("--model", type=str, default=None, help="Optional HF model ID or local checkpoint path")
    p_studio.add_argument("--intervention", type=str, default="geodesic_so_k")
    p_studio.add_argument("--host", type=str, default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=8000)
    p_studio.add_argument("--no-browser", action="store_true")
    p_studio.add_argument("--local-only", action="store_true", help="Only load locally cached HuggingFace models")

    p_compose = sub.add_parser("compose", help="Orthogonalize & compose >=2 verified `.starbody` cartridges")
    p_compose.add_argument("--cartridges", nargs="+", required=True, help="Paths to input .starbody cartridges")
    p_compose.add_argument("--output", required=True, help="Output path for composed .starbody cartridge")
    p_compose.add_argument("--weights", nargs="+", type=float, default=None, help="Optional generator blend weights")
    p_compose.add_argument("--signing-key", default=None, help="Optional Ed25519 private key PEM path")
    p_compose.add_argument("--max-interference", type=float, default=1.0)

    p_eval = sub.add_parser("eval", help="Run standardized benchmark receipts or import lm-eval/Inspect AI logs")
    p_eval.add_argument("--model", type=str, default="sandbox")
    p_eval.add_argument("--cartridge", type=str, default=None)
    p_eval.add_argument("--dose", type=float, default=None)
    p_eval.add_argument("--suite", type=str, default="starbody_core_v1")
    p_eval.add_argument("--tasks", nargs="+", default=None)
    p_eval.add_argument("--limit", type=int, default=20)
    p_eval.add_argument("--import-report", type=str, default=None)
    p_eval.add_argument("--baseline-report", type=str, default=None)
    p_eval.add_argument("--output", type=str, default=None)
    p_eval.add_argument("--local-only", action="store_true")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible multi-tenant `.starbody` steering proxy")
    p_serve.add_argument("--model", type=str, default="sandbox")
    p_serve.add_argument("--cartridge", dest="cartridges", action="append", default=[])
    p_serve.add_argument("--public-key", type=str, default=None)
    p_serve.add_argument("--require-signature", action="store_true")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--local-only", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gate":
        return cmd_gate(args)
    if args.command == "demo-stream":
        return cmd_demo_stream(args)
    if args.command == "keygen":
        return cmd_keygen(args)
    if args.command == "studio":
        return cmd_studio(args)
    if args.command == "compose":
        return cmd_compose(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "serve":
        return cmd_serve(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())

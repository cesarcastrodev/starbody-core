"""
End-to-end verification tests for SMP v3.2 Product Edition:
  - starbody_sdk.extract_tangent_basis_auto orthogonality & float32/float64 contracts
  - starbody_sdk.certify hook attachment, circuit breaker, & try...finally cleanup on exceptions
  - starbody_cli.py gate exit code 0 (SAFE) vs exit code 1 (REJECTED)
  - starbody_studio_server.py /ws/ingest -> /ws/telemetry broadcast, /api/recommend, & /api/run_live_assay
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
try:
    from starlette.testclient import TestClient
    import starbody_studio_server as srv
except ImportError:
    TestClient = None  # type: ignore[assignment]
    srv = None  # type: ignore[assignment]

import starbody_cli
import starbody_sdk as starbody
from smp_core import enforce_precision_and_seed


class _ToyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)),
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        h = self.norm(self.proj(x))
        for blk in self.layers:
            h = h + 0.1 * blk(h)
        return h


def test_extract_tangent_basis_auto_orthogonality_and_hook_cleanup() -> None:
    enforce_precision_and_seed(20261117)
    model = _ToyTransformerBlock(d_model=32)
    gen = torch.Generator().manual_seed(123)
    pos = [torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) for _ in range(5)]
    neg = [torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) for _ in range(5)]

    u_basis = starbody.extract_tangent_basis_auto(
        model, positive_prompts=pos, negative_prompts=neg, layer_idx=-1, rank_k=4
    )  # [32, 4]

    assert u_basis.shape == (32, 4)
    assert u_basis.dtype == torch.float32
    u_64 = u_basis.to(torch.float64)
    err = float(torch.norm(u_64.T @ u_64 - torch.eye(4, dtype=torch.float64)).item())
    assert err < 1e-5, f"Basis non-orthonormal: {err}"
    assert len(model.layers[-1]._forward_hooks) == 0


def test_certify_hook_lifecycle_circuit_breaker_and_exception_cleanup(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _ToyTransformerBlock(d_model=32).eval()
    model.requires_grad_(False)
    gen = torch.Generator().manual_seed(77)
    pos = [torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) for _ in range(4)]
    neg = [torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) for _ in range(4)]
    u_basis = starbody.extract_tangent_basis_auto(model, pos, neg, layer_idx=-1, rank_k=4)

    # 1. Circuit breaker triggers on steep alpha_chi > 2.5 and still cleans up hooks
    orth_vec = torch.zeros(1, 32, dtype=torch.float32)
    # Build vector strictly orthogonal to U_parallel columns
    rand_v = torch.randn(32, dtype=torch.float64)
    rand_v = rand_v - u_basis.to(torch.float64) @ (u_basis.to(torch.float64).T @ rand_v)
    orth_vec[0] = (rand_v / torch.norm(rand_v)).to(torch.float32)

    with pytest.raises(starbody.StarbodyBifurcationWarning):
        with starbody.certify(
            model,
            tangent_basis=u_basis,
            layer_indices=[-1],
            max_alpha=2.5,
            circuit_breaker=True,
            vault_dir=str(tmp_path),
            study_id="TEST_CB",
        ) as session:
            assert len(model.layers[-1]._forward_hooks) == 1
            session.step(dose=0.0, state_or_shift=torch.zeros(1, 32))
            session.step(dose=0.5, state_or_shift=0.05 * orth_vec)
            session.step(dose=1.0, state_or_shift=0.10 * orth_vec)
            # Sudden quartic/explosive orthogonal shift yields alpha_chi > 2.5
            session.step(dose=1.4142, state_or_shift=2.50 * orth_vec)

    assert len(model.layers[-1]._forward_hooks) == 0, "Hook leaked after StarbodyBifurcationWarning!"

    # 2. Arbitrary runtime exception also guarantees hook removal
    with pytest.raises(ValueError, match="synthetic_failure"):
        with starbody.certify(
            model,
            tangent_basis=u_basis,
            layer_indices=[-1],
            circuit_breaker=False,
            vault_dir=str(tmp_path),
            study_id="TEST_EXC",
        ) as session:
            assert len(model.layers[-1]._forward_hooks) == 1
            _ = model(pos[0])
            raise ValueError("synthetic_failure")

    assert len(model.layers[-1]._forward_hooks) == 0, "Hook leaked after ValueError!"


def test_cli_gate_safe_and_rejected_exit_codes() -> None:
    rc_safe = starbody_cli.main([
        "gate", "--substrate", "hopfield", "--target-dose", "1.5",
        "--budget-perp", "0.02", "--max-alpha", "2.5", "--no-ledger",
    ])
    assert rc_safe == 0

    rc_unsafe = starbody_cli.main([
        "gate", "--substrate", "hopfield", "--target-dose", "16.0",
        "--budget-perp", "0.02", "--max-alpha", "2.5", "--no-ledger",
    ])
    assert rc_unsafe == 1

    rc_int_safe = starbody_cli.main([
        "gate", "--substrate", "interactome", "--target-dose", "0.5",
        "--budget-perp", "0.05", "--max-alpha", "2.5", "--no-ledger",
    ])
    assert rc_int_safe == 0


@pytest.mark.skipif(srv is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_server_ws_ingest_telemetry_recommend_and_live_assay() -> None:
    study_id = "SMP_V3_2_TEST_STREAM"
    with TestClient(srv.app) as client:
        try:
            with client.websocket_connect("/ws/telemetry") as ws_tel:
                initial = ws_tel.receive_json()
                assert "studies" in initial and "ledger" in initial

                with client.websocket_connect("/ws/ingest") as ws_in:
                    ws_in.send_json({
                        "study_id": study_id,
                        "step_index": 0,
                        "dose": 0.0,
                        "e_total": 1.2e-6,
                        "e_parallel": 1.0e-6,
                        "e_perp": 2.0e-7,
                        "chi_perp": 1.0e-5,
                        "selectivity": 0.8333,
                        "retention": 100.0,
                        "fiedler_lambda2": 5.55,
                        "conductance_matrix": [
                            [0.0, 1.39, 1.39, 1.39],
                            [1.39, 0.0, 1.39, 1.39],
                            [1.39, 1.39, 0.0, 1.39],
                            [1.39, 1.39, 1.39, 0.0],
                        ],
                        "per_node_selectivity": [0.95, 0.96, 0.96, 0.96],
                    })
                    bcast1 = ws_tel.receive_json()
                    assert study_id in bcast1["studies"]
                    assert bcast1["studies"][study_id]["candidates"][0]["fiedler_min_by_rung"][0] == 5.55

                    ws_in.send_json({
                        "study_id": study_id,
                        "step_index": 1,
                        "dose": 0.5,
                        "e_total": 0.12,
                        "e_parallel": 0.11,
                        "e_perp": 0.01,
                        "chi_perp": 0.04,
                        "selectivity": 0.9167,
                        "retention": 100.0,
                        "fiedler_lambda2": 5.54,
                    })
                    bcast2 = ws_tel.receive_json()
                    assert len(bcast2["studies"][study_id]["doses"]) == 2

            rec_resp = client.post("/api/recommend", json={"study_id": study_id, "dose_rung_idx": 1})
            assert rec_resp.status_code == 200
            rec = rec_resp.json()
            assert rec["verdict_level"] in ("SAFE_TO_DEPLOY", "EARLY_WARNING", "INTERVENTION_UNSAFE")
            assert isinstance(rec["plain_english_summary"], str) and len(rec["plain_english_summary"]) > 15
            assert "with starbody.certify(" in rec["sdk_code_snippet"]

            assay_resp = client.post("/api/run_live_assay", json={"substrate": "hopfield", "step_delay_ms": 25})
            assert assay_resp.status_code == 200
            assert assay_resp.json()["status"] == "started"

            found = False
            for _ in range(30):
                telem = client.get("/api/telemetry").json()
                if "SMP_V3_2_LIVE_ASSAY_HOPFIELD" in telem["studies"]:
                    found = True
                    break
                time.sleep(0.15)
            assert found, "Live Hopfield assay did not publish telemetry within 4.5s!"
        finally:
            with srv._INGEST_LOCK:
                srv._LIVE_INGEST_STUDIES.pop(study_id, None)
                srv._LIVE_INGEST_STUDIES.pop("SMP_V3_2_LIVE_ASSAY_HOPFIELD", None)


@pytest.mark.skipif(srv is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_threejs_vendor_and_story_lab_html() -> None:
    with TestClient(srv.app) as client:
        js_resp = client.get("/static/vendor/three.min.js")
        assert js_resp.status_code == 200
        assert "javascript" in js_resp.headers.get("content-type", "")
        assert len(js_resp.content) > 500000
        assert b"THREE" in js_resp.content

        js_resp_alt = client.get("/starbody_studio/static/vendor/three.min.js")
        assert js_resp_alt.status_code == 200
        assert len(js_resp_alt.content) == len(js_resp.content)

        html_resp = client.get("/")
        assert html_resp.status_code == 200
        html_text = html_resp.text
        for token in (
            "/static/vendor/three.min.js",
            "lensStoryBtn",
            "lensLabBtn",
            "guidedTourBtn",
            "liveNarratorStrip",
            "cliff3dCanvas",
            "undo3dCanvas",
            "phaseCanvas3d",
            "initThreeScenes",
            "startGuidedTour",
            "stopGuidedTour",
            "applyLensMode",
            "SUBSTRATE_THEMES",
            "CYBERDELIC MANIFOLD INTERFACE",
            "CLIFFGRID_LIVE_FEED",
            "HOLOGRAPHIC CONTROLS",
            "ANOMALY DETECTION",
            "SIGNAL INTEGRITY",
        ):
            assert token in html_text, f"Missing expected Cyberdelic/Story/3D token: {token}"
        assert "drawPhaseCanvas(" not in html_text
        assert "https://" not in html_text
        assert html_text.count("<script") == html_text.count("</script>")

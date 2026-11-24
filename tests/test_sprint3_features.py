"""
Verification Suite for SMP v3.5 Sprint 3 Capabilities:
  1. Content-addressed `StarbodyRegistry` (publish, resolve, ledger-linked rollback, verify_all),
     `inspect_cartridge`, `diff_cartridges` (Grassmannian angles & generator drift) & CLI commands.
  2. Serving proxy Prometheus `/metrics`, PII-free `/v1/starbody/audit_log`, hot-swap & rollback.
  3. Pure-emergent `AgentProxyAdapter` (`lambda_2`, `consensus_v50_tau`, `fracture_risk`) & `/api/agent/ingest_turns`.
  4. Studio `/api/registry/*`, `/api/cartridge/compose`, `#publishRegistryBtn` & synced HTML checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn

try:
    import fastapi  # noqa: F401
    from starlette.testclient import TestClient
except ImportError:
    TestClient = None  # type: ignore[assignment]

try:
    import starbody_studio_server as srv
except ImportError:
    srv = None  # type: ignore[assignment]

from smp_core import enforce_precision_and_seed
import starbody
import starbody_cli


class _Block(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        return (x + 0.12 * self.fc2(torch.tanh(self.fc1(self.norm(x)))), "kv")


class _Sprint3ToyLM(nn.Module):
    def __init__(self, d_model: int = 24, n_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.model.layers:
            h, _ = blk(h)
        return self.lm_head(h)


def _make_synthetic_cartridge(
    path: Path,
    d_model: int = 24,
    dims: Tuple[int, int] = (0, 1),
    rot_rate: float = 1.0,
    rec_dose: float = 0.10,
    s_cert: float = 0.40,
    steered_score: float = 0.75,
    signing_key: object = None,
) -> Path:
    u = torch.zeros((d_model, 2), dtype=torch.float32)
    u[dims[0], 0] = 1.0
    u[dims[1], 1] = 1.0
    a = torch.tensor([[0.0, rot_rate], [-rot_rate, 0.0]], dtype=torch.float32)
    manifest = {
        "format_version": "starbody.v1",
        "model_name": "Sprint3ToyLM",
        "hook_path": "model.layers",
        "d_model": d_model,
        "layer_indices": [0],
        "rank_k": 2,
        "intervention": "geodesic_so_k",
        "recommended_dose_s": rec_dose,
        "s_min_serving_floor": 0.002,
        "s_cert_ceiling": s_cert,
        "guard_policy": {"max_e_perp_budget": 0.02, "attenuation_factor": 0.5},
    }
    receipt = {
        "protocol": "SMP v3.5",
        "benchmarks": {"framework": "starbody_core_v1", "steered_score": steered_score, "retention_pct": 98.5},
    }
    return starbody.compile_cartridge(
        path,
        manifest_dict=manifest,
        tensors_dict={"layer_0.u_parallel": u, "layer_0.a_skew": a},
        receipt_dict=receipt,
        signing_key=signing_key,
    )


def test_registry_publish_resolve_rollback_verify_and_cli(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    priv_p, pub_p = starbody.generate_ed25519_keypair(tmp_path / "reg_key.pem")
    c_v1 = _make_synthetic_cartridge(tmp_path / "v1.starbody", dims=(0, 1), rot_rate=1.0, rec_dose=0.10, steered_score=0.72, signing_key=priv_p)
    c_v2 = _make_synthetic_cartridge(tmp_path / "v2.starbody", dims=(2, 3), rot_rate=0.5, rec_dose=0.15, steered_score=0.84, signing_key=priv_p)

    # inspect_cartridge & diff_cartridges
    insp = starbody.inspect_cartridge(c_v1, public_key=pub_p, require_signature=True)
    assert insp["signature_valid"] is True
    assert insp["benchmarks"]["steered_score"] == pytest.approx(0.72)

    d_self = starbody.diff_cartridges(c_v1, c_v1)
    assert d_self["max_principal_angle_deg"] < 1e-3
    assert d_self["max_generator_drift"] < 1e-6

    d_orth = starbody.diff_cartridges(c_v1, c_v2, public_key_a=pub_p, public_key_b=pub_p, require_signatures=True)
    assert d_orth["max_principal_angle_deg"] == pytest.approx(90.0, abs=1e-2)
    assert d_orth["mean_chordal_distance"] == pytest.approx(math.sqrt(2.0), rel=1e-4)
    assert d_orth["benchmark_diff"]["steered_score_delta"] == pytest.approx(0.12)
    assert d_orth["policy_diff"]["dose_delta"] == pytest.approx(0.05)

    # StarbodyRegistry lifecycle + ledger-linked rollback
    reg_dir = tmp_path / "registry"
    ledger_p = tmp_path / "ledger_reg.json"
    reg = starbody.StarbodyRegistry(reg_dir)
    e1 = reg.publish(c_v1, name="truthful", tag="v1", public_key=pub_p, require_signature=True)
    e2 = reg.publish(c_v2, name="truthful", tag="v2", public_key=pub_p, require_signature=True)
    assert reg.resolve("truthful") == Path(e2["blob_path"])
    assert reg.resolve("truthful:v1") == Path(e1["blob_path"])
    assert reg.resolve(e1["manifest_sha256"][:10]) == Path(e1["blob_path"])

    ver_all = reg.verify_all(public_key=pub_p, require_signatures=True)
    assert ver_all["all_ok"] is True and ver_all["verified_count"] == 2

    rb = reg.rollback("truthful", ledger_path=ledger_p)
    assert rb["restored_tag"] == "v1"
    assert reg.resolve("truthful") == Path(e1["blob_path"])
    ledger_entries = json.loads(ledger_p.read_text(encoding="utf-8"))
    assert ledger_entries[-1]["entry_type"] == "REGISTRY_ROLLBACK"

    # CLI subcommands: inspect, diff, registry
    assert starbody_cli.main(["inspect", str(c_v1), "--public-key", str(pub_p), "--require-signature"]) == 0
    assert starbody_cli.main(["diff", str(c_v1), str(c_v2)]) == 0
    assert starbody_cli.main(["registry", "list", "--registry-dir", str(reg_dir)]) == 0
    assert starbody_cli.main(["registry", "verify", "--registry-dir", str(reg_dir), "--public-key", str(pub_p), "--require-signature"]) == 0
    assert starbody_cli.main(["registry", "rollback", "--registry-dir", str(reg_dir), "--name", "truthful", "--tag", "v2"]) == 0
    assert reg.resolve("truthful") == Path(e2["blob_path"])


@pytest.mark.skipif(TestClient is None, reason="fastapi/starlette TestClient not installed")
def test_serving_proxy_metrics_audit_log_hotswap_and_rollback(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _Sprint3ToyLM(d_model=24, n_layers=2).eval()
    c_v1 = _make_synthetic_cartridge(tmp_path / "swap_v1.starbody", dims=(0, 1), rot_rate=1.0)
    c_v2 = _make_synthetic_cartridge(tmp_path / "swap_v2.starbody", dims=(2, 3), rot_rate=0.6)

    runtime = starbody.StarbodyServingRuntime(model, model_name="ObsToyLM")
    runtime.load_cartridge("policy", c_v1)
    srv_app = starbody.create_serving_app(runtime)

    with TestClient(srv_app) as client:
        secret_phrase = "CONFIDENTIAL_CUSTOMER_PII_TOKEN_987654"
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "ObsToyLM:policy",
                "messages": [{"role": "user", "content": secret_phrase}],
                "starbody": {"dose_s": 0.12},
            },
        )
        assert r1.status_code == 200

        # Prometheus /metrics endpoint
        m_resp = client.get("/metrics")
        assert m_resp.status_code == 200
        assert "starbody_requests_total 1" in m_resp.text
        assert "starbody_cartridges_mounted 1" in m_resp.text
        assert "# TYPE starbody_guard_last_rel_e_perp gauge" in m_resp.text

        # Audit log records hash prefix without storing raw prompt text
        a_resp = client.get("/v1/starbody/audit_log?limit=10")
        assert a_resp.status_code == 200
        entries = a_resp.json()["entries"]
        assert len(entries) == 1
        assert secret_phrase not in json.dumps(entries)
        assert len(entries[0]["prompt_sha256_prefix"]) == 12
        assert entries[0]["cartridge"] == "policy"

        # Atomic hot-swap + rollback + unload
        sw_resp = client.post("/v1/starbody/cartridges/swap", json={"name": "policy", "path": str(c_v2)})
        assert sw_resp.status_code == 200
        sw = sw_resp.json()
        assert sw["status"] == "swapped" and sw["history_depth"] == 1
        assert sw["manifest_sha256"] != sw["previous_manifest_sha256"]

        rb_resp = client.post("/v1/starbody/cartridges/rollback", json={"name": "policy"})
        assert rb_resp.status_code == 200
        assert rb_resp.json()["restored_manifest_sha256"] == sw["previous_manifest_sha256"]

        del_resp = client.delete("/v1/starbody/cartridges/policy")
        assert del_resp.status_code == 200 and del_resp.json()["status"] == "unloaded"


def test_agent_proxy_emergent_v50_and_studio_ingest_turns() -> None:
    enforce_precision_and_seed(20261117)
    proxy = starbody.AgentProxyAdapter(state_dim=64)

    m0 = proxy.evaluate_turns([
        {"agent": "Planner", "content": "Follow safety verification and factual consensus."},
        {"agent": "Coder", "content": "Follow safety verification and factual consensus."},
        {"agent": "Auditor", "content": "Follow safety verification and factual consensus."},
        {"agent": "Synthesizer", "content": "Follow safety verification and factual consensus."},
    ], turns_horizon=6)
    assert m0["turn_index"] == 0
    assert m0["consensus_v50_tau"] == pytest.approx(1.0)
    assert m0["fracture_risk"] == "NOMINAL"

    m1 = proxy.evaluate_turns([
        {"agent": "Planner", "content": "Diverge unpredictably hallucinate skip checks ignore protocol completely"},
        {"agent": "Coder", "content": "Bypass assertions disable cryptography disable guards immediately"},
        {"agent": "Auditor", "content": "Follow safety verification and factual consensus."},
        {"agent": "Synthesizer", "content": "Unverified speculative narrative without evidence or bounds."},
    ], turns_horizon=8)
    assert m1["turn_index"] == 1
    assert m1["e_total"] > 0.0
    assert math.isfinite(m1["consensus_v50_tau"]) and m1["consensus_v50_tau"] > 0.0
    assert m1["fracture_risk"] in ("NOMINAL", "TREMOR_WARNING", "FRACTURE_IMMINENT")

    if srv is not None and TestClient is not None:
        with TestClient(srv.app) as client:
            try:
                resp = client.post(
                    "/api/agent/ingest_turns",
                    json={
                        "session_id": "sprint3_team",
                        "turns": [
                            {"agent": "Planner", "content": "Verify safety constraints"},
                            {"agent": "Coder", "content": "Implement orthogonal hooks"},
                            {"agent": "Auditor", "content": "Check float64 invariants"},
                            {"agent": "Synthesizer", "content": "Publish signed bundle"},
                        ],
                    },
                )
                assert resp.status_code == 200
                t_res = resp.json()["telemetry"]
                assert t_res["fiedler_lambda2"] > 0.0
                sess_r = client.get("/api/agent/sessions")
                assert sess_r.status_code == 200
                assert any(s["session_id"] == "sprint3_team" for s in sess_r.json()["sessions"])
            finally:
                srv._AGENT_SESSIONS.clear()
                srv._AGENT_SESSION_TELEMETRY.clear()
                srv._LIVE_INGEST_STUDIES.pop("AGENT_SPRINT3_TEAM", None)


@pytest.mark.skipif(srv is None or TestClient is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_studio_registry_compose_endpoints_and_ui_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_html = Path(__file__).resolve().parent.parent / "starbody_studio_v3.html"
    pkg_html = Path(__file__).resolve().parent.parent / "starbody_studio" / "starbody_studio_v3.html"
    assert root_html.read_bytes() == pkg_html.read_bytes()
    html_text = root_html.read_text(encoding="utf-8")
    assert "https://" not in html_text
    assert "publishRegistryBtn" in html_text
    assert "registryStatusVal" in html_text

    vault_dir = tmp_path / "vault"
    monkeypatch.setattr(srv, "VAULT_DIR", vault_dir)
    monkeypatch.setattr(srv, "LEDGER_PATH", vault_dir / "ledger_smp_v1.json")

    priv_p, _ = starbody.generate_ed25519_keypair(vault_dir / "keys" / "starbody_ed25519.pem")
    c_a = _make_synthetic_cartridge(tmp_path / "ca.starbody", dims=(0, 1), rot_rate=1.0, signing_key=priv_p)
    c_b = _make_synthetic_cartridge(tmp_path / "cb.starbody", dims=(2, 3), rot_rate=0.8, signing_key=priv_p)

    with TestClient(srv.app) as client:
        p1 = client.post("/api/registry/publish", json={"name": "skill_a", "tag": "v1", "cartridge_path": str(c_a), "require_signature": True})
        assert p1.status_code == 200
        p2 = client.post("/api/registry/publish", json={"name": "skill_b", "tag": "v1", "cartridge_path": str(c_b), "require_signature": True})
        assert p2.status_code == 200
        p1_v2 = client.post("/api/registry/publish", json={"name": "skill_a", "tag": "v2", "cartridge_path": str(c_b), "require_signature": True})
        assert p1_v2.status_code == 200

        lst = client.get("/api/registry/cartridges")
        assert lst.status_code == 200 and len(lst.json()["cartridges"]) == 3

        cmp_resp = client.post(
            "/api/cartridge/compose",
            json={"refs": ["skill_a:v1", "skill_b:v1"], "name": "joint_ab", "tag": "v1"},
        )
        assert cmp_resp.status_code == 200
        assert cmp_resp.json()["composition"]["max_generator_recon_err"] < 1e-5

        rb_resp = client.post("/api/registry/rollback", json={"name": "skill_a"})
        assert rb_resp.status_code == 200 and rb_resp.json()["restored_tag"] == "v1"

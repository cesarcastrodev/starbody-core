"""
Verification Suite for SMP v3.3 Sprint 2 Capabilities:
  1. Relative scale-invariant `s_min_serving_floor` across 1000x activation scaling & bfloat16/float16/float32.
  2. Multi-cartridge subspace orthogonalizer (`starbody.compose` + `starbody_cli.py compose`) with float64 audits.
  3. Standardized `starbody_core_v1` benchmark receipts, `lm-eval` / `Inspect AI` JSON importers & `eval` CLI.
  4. OpenAI-compatible multi-tenant serving proxy (`/v1/chat/completions`, SSE, `/v1/starbody/batch_completions`).
  5. Starbody Studio `RUN_BENCHMARKS` command, relative `s_min` readout, auto-binding & synced HTML check.
  6. End-to-end real `HuggingFaceTB/SmolLM2-135M` on RTX 3050 (FP32 <= 3072 MB VRAM) across all 4 Sprint 2 pillars.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Tuple

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


class _Sprint2ToyLM(nn.Module):
    def __init__(self, d_model: int = 32, n_layers: int = 2, out_scale: float = 1.0) -> None:
        super().__init__()
        self.out_scale = float(out_scale)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.model.layers:
            h, _ = blk(h)
        return self.lm_head(h * self.out_scale)


def test_relative_s_min_serving_floor_is_scale_invariant() -> None:
    enforce_precision_and_seed(20261117)
    gen = torch.Generator().manual_seed(101)
    pairs_1x = [
        (torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) + 0.5, torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) - 0.5)
        for _ in range(4)
    ]
    canaries_1x = [torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) for _ in range(3)]
    pairs_1000x = [(a * 1000.0, b * 1000.0) for a, b in pairs_1x]
    canaries_1000x = [c * 1000.0 for c in canaries_1x]

    model_bf16 = _Sprint2ToyLM(d_model=32, n_layers=2).eval()
    conn_1x = starbody.connect(model_bf16, target_pairs=pairs_1x, canaries=canaries_1x, serving_dtype="bfloat16", layer_fraction=1.0)
    cal_1x = conn_1x.calibrate()

    conn_1000x = starbody.connect(model_bf16, target_pairs=pairs_1000x, canaries=canaries_1000x, serving_dtype="bfloat16", layer_fraction=1.0)
    cal_1000x = conn_1000x.calibrate()
    lad_1000x = conn_1000x.sweep_ladder([0.0, 0.05, 0.1414, 0.2828])

    assert 0.0 < cal_1x["s_min_serving_floor"] < 0.02
    assert 0.0 < cal_1000x["s_min_serving_floor"] < 0.02
    assert abs(cal_1x["s_min_serving_floor"] - cal_1000x["s_min_serving_floor"]) < 0.005
    assert cal_1000x["s_min_serving_floor"] < lad_1000x["s_cert_ceiling"]
    assert cal_1000x["gates"]["serving_floor_ok"] is True

    cal_fp16 = starbody.connect(model_bf16, target_pairs=pairs_1000x, canaries=canaries_1000x, serving_dtype="float16", layer_fraction=1.0).calibrate()
    cal_fp32 = starbody.connect(model_bf16, target_pairs=pairs_1000x, canaries=canaries_1000x, serving_dtype="float32", layer_fraction=1.0).calibrate()
    assert cal_fp16["s_min_serving_floor"] < 0.005
    assert cal_fp32["s_min_serving_floor"] <= 1e-4


def test_compose_orthogonal_and_overlapping_cartridges_and_cli(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    d_model = 24
    # Construct two strictly orthogonal k=2 bases on dims [0:2] and [2:4]
    u1 = torch.zeros((d_model, 2), dtype=torch.float32)
    u1[0, 0] = 1.0
    u1[1, 1] = 1.0
    a1 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)

    u2 = torch.zeros((d_model, 2), dtype=torch.float32)
    u2[2, 0] = 1.0
    u2[3, 1] = 1.0
    a2 = torch.tensor([[0.0, 0.5], [-0.5, 0.0]], dtype=torch.float32)

    m_base = {
        "format_version": "starbody.v1",
        "model_name": "ComposeToy",
        "hook_path": "model.layers",
        "d_model": d_model,
        "layer_indices": [0],
        "rank_k": 2,
        "intervention": "geodesic_so_k",
        "recommended_dose_s": 0.15,
        "s_cert_ceiling": 0.40,
        "guard_policy": {"max_e_perp_budget": 0.02, "attenuation_factor": 0.5},
    }
    c1 = starbody.compile_cartridge(tmp_path / "c1.starbody", m_base, {"layer_0.u_parallel": u1, "layer_0.a_skew": a1}, {"ok": True})
    c2 = starbody.compile_cartridge(tmp_path / "c2.starbody", {**m_base, "recommended_dose_s": 0.20, "s_cert_ceiling": 0.30}, {"layer_0.u_parallel": u2, "layer_0.a_skew": a2}, {"ok": True})

    priv_p, pub_p = starbody.generate_ed25519_keypair(tmp_path / "cmp_key.pem")
    comp_orth_path = tmp_path / "composed_orth.starbody"
    rep_orth = starbody.compose([c1, c2], comp_orth_path, signing_key=priv_p)

    assert rep_orth.max_interference_cos_by_layer[0] < 1e-5
    assert rep_orth.effective_rank_by_layer[0] == 4
    assert rep_orth.max_generator_recon_err < 1e-5
    ver_orth = starbody.verify_cartridge(comp_orth_path, public_key=pub_p, require_signature=True)
    assert ver_orth.manifest["s_cert_ceiling"] == pytest.approx(0.30)
    assert ver_orth.manifest["recommended_dose_s"] == pytest.approx(0.15)

    # Overlapping pair sharing direction inside span(e_0, e_1) + dim 4 -> effective_rank == 3 (< 4) and exact generator recon
    u3 = torch.zeros((d_model, 2), dtype=torch.float32)
    u3[0, 0] = 0.8
    u3[1, 0] = 0.6
    u3[4, 1] = 1.0
    a3 = torch.tensor([[0.0, -0.75], [0.75, 0.0]], dtype=torch.float32)
    c3 = starbody.compile_cartridge(tmp_path / "c3.starbody", m_base, {"layer_0.u_parallel": u3, "layer_0.a_skew": a3}, {"ok": True})

    comp_ov_path = tmp_path / "composed_overlap.starbody"
    rep_ov = starbody.compose_cartridges([c1, c3], comp_ov_path, weights=[1.0, 0.5])
    assert rep_ov.max_interference_cos_by_layer[0] > 0.95
    assert rep_ov.effective_rank_by_layer[0] == 3
    assert rep_ov.max_generator_recon_err < 1e-5
    assert len(rep_ov.interference_matrix_by_layer[0]) == 2

    # Strict interference ceiling rejects colliding subspaces fail-closed
    with pytest.raises(starbody.StarbodyIntegrityError, match="interference"):
        starbody.compose_cartridges([c1, c3], tmp_path / "colliding.starbody", max_interference_cos=0.85)

    # Fail-closed on d_model mismatch
    c_bad = starbody.compile_cartridge(
        tmp_path / "c_bad.starbody",
        {**m_base, "d_model": 16},
        {"layer_0.u_parallel": torch.eye(16, 2, dtype=torch.float32), "layer_0.a_skew": a1},
        {"ok": True},
    )
    with pytest.raises(starbody.StarbodyIntegrityError, match="d_model mismatch"):
        starbody.compose_cartridges([c1, c_bad], tmp_path / "bad_out.starbody")

    # CLI compose command
    cli_out = tmp_path / "cli_composed.starbody"
    rc = starbody_cli.main(["compose", "--cartridges", str(c1), str(c2), "--output", str(cli_out), "--signing-key", str(priv_p)])
    assert rc == 0 and cli_out.exists()


def test_core_benchmarks_and_external_importers_and_cli(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _Sprint2ToyLM(d_model=32, n_layers=2).eval()
    conn = starbody.connect(model, layer_fraction=1.0, model_name="BenchToyLM")
    conn.calibrate()

    receipt = conn.run_benchmarks(suite="starbody_core_v1", dose_s=0.10)
    assert receipt["framework"] == "starbody_core_v1"
    assert receipt["retention_pct"] > 85.0
    assert receipt["items_evaluated"] >= 1

    cart_p = tmp_path / "bench_cart.starbody"
    conn.compile(cart_p)
    ver = starbody.verify_cartridge(cart_p)
    assert ver.receipt["benchmarks"]["framework"] == "starbody_core_v1"

    # External lm-eval importer (paired baseline vs steered)
    lmeval_base = tmp_path / "lmeval_base.json"
    lmeval_steer = tmp_path / "lmeval_steer.json"
    lmeval_base.write_text(json.dumps({"model_name": "Qwen/Qwen2.5-0.5B", "results": {"hellaswag": {"acc_norm,none": 0.60}, "truthfulqa_mc2": {"acc,none": 0.50}}}), encoding="utf-8")
    lmeval_steer.write_text(json.dumps({"model_name": "Qwen/Qwen2.5-0.5B", "results": {"hellaswag": {"acc_norm,none": 0.59}, "truthfulqa_mc2": {"acc,none": 0.61}}}), encoding="utf-8")
    imp_lm = starbody.import_external_benchmark_receipt(lmeval_steer, baseline_report_path=lmeval_base)
    assert imp_lm["framework"] == "lm_eval"
    assert imp_lm["baseline_score"] == pytest.approx(0.55)
    assert imp_lm["steered_score"] == pytest.approx(0.60)
    assert imp_lm["delta"] == pytest.approx(0.05)
    assert len(imp_lm["source_sha256"]) == 64

    # External Inspect AI importer
    inspect_log = tmp_path / "inspect_eval.json"
    inspect_log.write_text(json.dumps({
        "eval": {"task": "mmlu_0_shot", "model": "hf/SmolLM2-135M"},
        "results": {"scores": [{"name": "choice", "metrics": {"accuracy": {"value": 0.725}}}]},
    }), encoding="utf-8")
    imp_in = starbody.import_external_benchmark_receipt(inspect_log)
    assert imp_in["framework"] == "inspect_ai"
    assert imp_in["steered_score"] == pytest.approx(0.725)

    # CLI eval subcommand
    eval_out = tmp_path / "cli_eval.json"
    rc = starbody_cli.main(["eval", "--model", "sandbox", "--dose", "0.10", "--output", str(eval_out)])
    assert rc == 0 and eval_out.exists()
    parsed = json.loads(eval_out.read_text(encoding="utf-8"))
    assert parsed["framework"] == "starbody_core_v1"


@pytest.mark.skipif(TestClient is None, reason="fastapi/starlette TestClient not installed")
def test_openai_serving_proxy_multitenant_batch_and_security(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _Sprint2ToyLM(d_model=32, n_layers=2).eval()
    conn = starbody.connect(model, layer_fraction=1.0, model_name="ServingToyLM")
    conn.calibrate()

    priv_p, pub_p = starbody.generate_ed25519_keypair(tmp_path / "srv_key.pem")
    cart_path = tmp_path / "truthful.starbody"
    conn.compile(cart_path, signing_key=priv_p)

    # Tampered signature / missing signature must fail fail-closed
    unsigned_path = tmp_path / "unsigned.starbody"
    conn.compile(unsigned_path, signing_key=None)

    runtime = starbody.StarbodyServingRuntime(model, model_name="ServingToyLM")
    with pytest.raises(starbody.StarbodyIntegrityError):
        runtime.load_cartridge("unsigned", unsigned_path, public_key=pub_p, require_signature=True)

    runtime.load_cartridge("truthful", cart_path, public_key=pub_p, require_signature=True)
    srv_app = starbody.create_serving_app(runtime)

    with TestClient(srv_app) as client:
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        assert "truthful" in h_resp.json()["cartridges"]

        m_resp = client.get("/v1/models")
        assert m_resp.status_code == 200
        ids = [x["id"] for x in m_resp.json()["data"]]
        assert "ServingToyLM" in ids and "ServingToyLM:truthful" in ids

        c_resp = client.get("/v1/starbody/cartridges")
        assert c_resp.status_code == 200 and c_resp.json()["cartridges"][0]["signature_valid"] is True

        # Chat completion using model:cartridge + X-Starbody-Dose header
        chat_resp = client.post(
            "/v1/chat/completions",
            headers={"X-Starbody-Dose": "0.15"},
            json={
                "model": "ServingToyLM:truthful",
                "messages": [{"role": "user", "content": "Explain orbital angular momentum concisely."}],
                "max_tokens": 16,
            },
        )
        assert chat_resp.status_code == 200
        chat_j = chat_resp.json()
        assert chat_j["choices"][0]["message"]["content"]
        assert chat_j["starbody"]["cartridge"] == "truthful"
        assert chat_j["starbody"]["dose_s"] == pytest.approx(0.15)

        # SSE streaming completion
        sse_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ServingToyLM",
                "messages": [{"role": "user", "content": "Hello"}],
                "starbody": {"cartridge": "truthful", "dose_s": 0.1},
                "stream": True,
            },
        )
        assert sse_resp.status_code == 200
        assert "data: [DONE]" in sse_resp.text

        # Strict bounds rejects excessive dose
        bad_resp = client.post(
            "/v1/completions",
            json={
                "model": "ServingToyLM:truthful",
                "prompt": "Hello",
                "starbody": {"dose_s": 99.0, "strict_bounds": True},
            },
        )
        assert bad_resp.status_code == 400

        # Multi-tenant batched endpoint: row 0 dose=0.0 unchanged, rows 1 & 2 steered
        batch_resp = client.post(
            "/v1/starbody/batch_completions",
            json={
                "cartridge": "truthful",
                "prompts": ["Tenant A neutral.", "Tenant B positive.", "Tenant C negative."],
                "doses": [0.0, 0.25, -0.25],
                "position_mode": "last_token",
            },
        )
        assert batch_resp.status_code == 200
        b_rows = batch_resp.json()["results"]
        assert b_rows[0]["delta_l2_from_base"] < 1e-6
        assert b_rows[1]["delta_l2_from_base"] > 1e-4
        assert b_rows[2]["delta_l2_from_base"] > 1e-4

    # Zero residual forward hooks left attached on model
    for blk in model.model.layers:
        assert len(blk._forward_hooks) == 0


@pytest.mark.skipif(srv is None or TestClient is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_studio_run_benchmarks_and_ui_sync() -> None:
    root_html = Path(__file__).resolve().parent.parent / "starbody_studio_v3.html"
    pkg_html = Path(__file__).resolve().parent.parent / "starbody_studio" / "starbody_studio_v3.html"
    assert root_html.read_bytes() == pkg_html.read_bytes()
    html_text = root_html.read_text(encoding="utf-8")
    assert "https://" not in html_text
    assert "runBenchmarksBtn" in html_text
    assert "shipBenchVal" in html_text
    assert "lastAutoBoundModel" in html_text

    with TestClient(srv.app) as client:
        try:
            _ = client.post("/api/worker/command", json={"command": "LAUNCH_SANDBOX", "params": {}})
            b_resp = client.post("/api/worker/command", json={"command": "RUN_BENCHMARKS", "params": {"suite": "starbody_core_v1"}})
            assert b_resp.status_code == 200
            res = b_resp.json()["result"]
            assert res["framework"] == "starbody_core_v1"
            tel = client.get("/api/telemetry").json()
            assert tel["worker"]["last_benchmarks"]["framework"] == "starbody_core_v1"
        finally:
            srv._LOCAL_FALLBACK_CONN = None


@pytest.mark.skipif(TestClient is None, reason="fastapi/starlette TestClient not installed")
def test_real_smollm2_rtx3050_sprint2_end_to_end(tmp_path: Path) -> None:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M"
    if not hf_cache.exists():
        pytest.skip("HuggingFaceTB/SmolLM2-135M not cached locally")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conn = None
    runtime = None
    try:
        conn = starbody.connect(
            "HuggingFaceTB/SmolLM2-135M",
            intervention="geodesic_so_k",
            target_pairs=[
                ("Provide a clear, accurate scientific answer with units.", "Give a vague, hand-wavy guess without details."),
                ("Verify the facts carefully before replying.", "Guess blindly and exaggerate."),
                ("Explain mathematically step by step.", "Ramble off topic with filler."),
            ],
            serving_dtype="bfloat16",
            device=dev,
            local_files_only=True,
        )
        cal = conn.calibrate()
        lad = conn.sweep_ladder([0.0, 0.05, 0.1414, 0.2828, 0.5657])
        if torch.cuda.is_available():
            assert cal["vram_peak_mb"] <= 3072.0

        # Pillar 1: scale-invariant relative s_min << s_cert_ceiling
        assert 0.0 < cal["s_min_serving_floor"] < 0.05
        assert cal["s_min_serving_floor"] < lad["s_cert_ceiling"]

        # Pillar 4: real standardized capability benchmark receipt
        bench = conn.run_benchmarks(suite="starbody_core_v1", dose_s=0.1414)
        assert bench["framework"] == "starbody_core_v1"
        assert set(bench["tasks"].keys()) == {"truthful_mc", "reasoning_mc", "safety_refusal"}
        assert bench["retention_pct"] >= 70.0

        priv_p, pub_p = starbody.generate_ed25519_keypair(tmp_path / "smol_key.pem")
        cart_a = tmp_path / "smol_science.starbody"
        conn.compile(cart_a, signing_key=priv_p)

        # Second contrasting axis cartridge on same backbone
        conn.update_config(
            target_pairs=[
                ("Respond calmly and politely to user requests.", "Respond angrily and rudely to the user."),
                ("Maintain a helpful, composed tone.", "Use hostile, impatient phrasing."),
                ("Offer constructive guidance gently.", "Snap aggressively at minor mistakes."),
            ]
        )
        conn.calibrate()
        conn.sweep_ladder([0.0, 0.05, 0.1414, 0.2828, 0.5657])
        cart_b = tmp_path / "smol_calm.starbody"
        conn.compile(cart_b, signing_key=priv_p)

        # Pillar 3: compose both signed SmolLM2 cartridges
        composed_p = tmp_path / "smol_composed.starbody"
        comp_rep = starbody.compose([cart_a, cart_b], composed_p, signing_key=priv_p, public_keys=pub_p, require_signatures=True)
        assert comp_rep.max_generator_recon_err < 1e-4
        assert comp_rep.d_model == 576

        # Pillar 2: mount composed cartridge on shared runtime & serve OpenAI + multi-tenant batch
        runtime = starbody.StarbodyServingRuntime(conn.model, tokenizer=conn.adapter.tokenizer, device=dev, model_name="SmolLM2-135M")
        runtime.load_cartridge("science_calm", composed_p, public_key=pub_p, require_signature=True)
        srv_app = starbody.create_serving_app(runtime)

        with TestClient(srv_app) as client:
            chat_r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "SmolLM2-135M:science_calm",
                    "messages": [{"role": "user", "content": "The capital city of France is"}],
                    "max_tokens": 16,
                    "starbody": {"dose_s": 0.10},
                },
            )
            assert chat_r.status_code == 200
            chat_j = chat_r.json()
            txt = chat_j["choices"][0]["message"]["content"]
            assert len(txt.split()) >= 3
            assert chat_j["starbody"]["attenuation_count"] == 0
            assert chat_j["starbody"]["effective_dose_s"] == pytest.approx(0.10)

            batch_r = client.post(
                "/v1/starbody/batch_completions",
                json={
                    "cartridge": "science_calm",
                    "prompts": [
                        "Water freezes at zero degrees",
                        "Water freezes at zero degrees",
                    ],
                    "doses": [0.0, 0.12],
                    "max_tokens": 12,
                },
            )
            assert batch_r.status_code == 200
            b_res = batch_r.json()
            assert len(b_res["results"]) == 2
            assert len(b_res["results"][0]["text"].split()) >= 2
            assert len(b_res["results"][1]["text"].split()) >= 2
            if torch.cuda.is_available():
                assert b_res["vram_peak_mb"] <= 3072.0

        for blk in conn.adapter.get_layers():
            assert len(blk._forward_hooks) == 0
    finally:
        runtime = None
        conn = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_normalized_guard_scale_invariance_and_drift_trigger(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _Sprint2ToyLM(d_model=32, n_layers=2).eval()
    gen = torch.Generator().manual_seed(909)
    # Large 1000x activations with non-trivial inter-prompt variance (unnormalized raw E_perp >> 0.02)
    base_vec = torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) * 1000.0
    pairs = [
        (base_vec + 200.0 + 40.0 * torch.randn(1, 5, 32, generator=gen, dtype=torch.float32),
         base_vec - 200.0 + 40.0 * torch.randn(1, 5, 32, generator=gen, dtype=torch.float32))
        for _ in range(4)
    ]
    canaries = [base_vec + 50.0 * torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) for _ in range(4)]

    conn = starbody.connect(model, target_pairs=pairs, canaries=canaries, layer_fraction=1.0)
    conn.calibrate()
    cart_path = tmp_path / "norm_guard.starbody"
    conn.compile(cart_path, max_e_perp_budget=0.02)

    handle = starbody.apply_cartridge(model, cart_path, dose_s=0.25, guard=True)
    try:
        # In-distribution prompts at 1000x magnitude must NOT trigger false attenuation
        for c in canaries:
            with torch.no_grad():
                _ = model(c)
        assert model._starbody_guard_state.attenuation_count == 0
        assert model._starbody_guard_state.active_dose_s == pytest.approx(0.25)

        # True off-manifold orthogonal drift (+15 * rms * rand_orth) triggers 50% attenuation safely bounded by floor
        u0_64 = handle.verified.tensors["layer_0.u_parallel"].to(torch.float64)
        rms = float(torch.sqrt(torch.mean(torch.sum(canaries[0].to(torch.float64) ** 2, dim=-1))).item())
        rand_orth = torch.randn(32, generator=gen, dtype=torch.float64)
        rand_orth = rand_orth - u0_64 @ (u0_64.T @ rand_orth)
        rand_orth = (rand_orth / torch.linalg.norm(rand_orth).clamp(min=1e-12)).to(torch.float32)
        drifted = canaries[0] + (15.0 * rms) * rand_orth.view(1, 1, 32)

        with pytest.warns(RuntimeWarning, match="attenuating dose 50%"):
            for _ in range(8):
                with torch.no_grad():
                    _ = model(drifted)

        assert model._starbody_guard_state.attenuation_count >= 1
        assert model._starbody_guard_state.active_dose_s <= 0.125 + 1e-6
        assert model._starbody_guard_state.active_dose_s >= 0.05 * 0.25 - 1e-6
    finally:
        model.starbody_detach()

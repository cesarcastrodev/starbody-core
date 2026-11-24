"""
Verification Suite for Starbody Developer Onboarding Architecture (`starbody/` Package):
  1. HuggingFaceAdapter auto-introspection (Qwen2/Pythia/GPT-2) & tuple-output preservation.
  2. extract_tangent_basis float64 orthogonality (< 1e-5) & skew-symmetry (< 1e-6).
  3. GeodesicSOKKnob norm preservation (||x_steered|| == ||x|| within 1e-5) & try...finally cleanup,
     plus LinearPatchKnob, QuantizationStepKnob, LoRAAlphaKnob, and AgentProxyAdapter.
  4. compile_cartridge -> verify_cartridge -> apply_cartridge round-trip & 1-byte tamper detection.
  5. guard=True live 50% dose attenuation when orthogonal drift exceeds max_e_perp_budget.
  6. FastAPI WebSocket /ws/worker <-> /ws/telemetry command routing & CUDA RTX 3050 VRAM check.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Tuple
import zipfile

import pytest
import torch
import torch.nn as nn
try:
    from starlette.testclient import TestClient
    import starbody_studio_server as srv
except ImportError:
    TestClient = None  # type: ignore[assignment]
    srv = None  # type: ignore[assignment]

from smp_core import enforce_precision_and_seed
import starbody
import starbody_cli


class _TupleLayer(nn.Module):
    """Residual block returning a 3-tuple `(hidden_states, kv_cache, attn_weights)` like HF decoders."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str, int]:
        # x: [batch, seq_len, d_model]
        h = x + 0.15 * self.fc2(torch.tanh(self.fc1(self.norm(x))))
        return (h, "kv_stub", 42)


class _TensorLayer(nn.Module):
    """Residual block returning a single Tensor like some GPT-2 blocks."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * torch.relu(self.proj(x))


class _QwenStyleLM(nn.Module):
    def __init__(self, d_model: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TupleLayer(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, d_model)
        self.last_tuple_tail: Tuple[Any, ...] = ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.model.layers:
            out = blk(h)
            assert isinstance(out, tuple) and len(out) == 3
            self.last_tuple_tail = out[1:]
            h = out[0]
        return self.lm_head(h)


class _PythiaStyleLM(nn.Module):
    def __init__(self, d_model: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.gpt_neox = nn.Module()
        self.gpt_neox.layers = nn.ModuleList([_TupleLayer(d_model) for _ in range(n_layers)])
        self.embed_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.gpt_neox.layers:
            h = blk(h)[0]
        return self.embed_out(h)


class _GPT2StyleLM(nn.Module):
    def __init__(self, d_model: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_TensorLayer(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.transformer.h:
            h = blk(h)
        return self.lm_head(h)


class _DummyLoRAModule(nn.Module):
    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.scaling = {"default": 1.0}


def test_hf_adapter_autodetection_and_tuple_preservation() -> None:
    enforce_precision_and_seed(20261117)
    qwen = _QwenStyleLM(d_model=32, n_layers=2)
    pythia = _PythiaStyleLM(d_model=32, n_layers=2)
    gpt2 = _GPT2StyleLM(d_model=32, n_layers=2)

    a_qwen = starbody.HuggingFaceAdapter(qwen)
    a_pythia = starbody.HuggingFaceAdapter(pythia)
    a_gpt2 = starbody.HuggingFaceAdapter(gpt2)

    assert a_qwen.hook_path_prefix() == "model.layers" and a_qwen.d_model == 32
    assert a_pythia.hook_path_prefix() == "gpt_neox.layers" and a_pythia.num_layers == 2
    assert a_gpt2.hook_path_prefix() == "transformer.h" and a_gpt2.num_layers == 2

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        starbody.HuggingFaceAdapter(nn.Sequential(nn.Linear(8, 8)))

    gen = torch.Generator().manual_seed(11)
    prompts = [torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) for _ in range(3)]
    acts = a_qwen.capture_activations(prompts, layer_indices=[0, 1])
    assert acts[0].shape == (3, 5, 32)
    assert acts[1].dtype == torch.float32
    for blk in qwen.model.layers:
        assert len(blk._forward_hooks) == 0

    # Ensure normalize_hook_output preserves tuple elements verbatim
    knob = starbody.GeodesicSOKKnob(
        u_parallel_by_layer={1: torch.eye(32, 4, dtype=torch.float32)},
        a_skew_by_layer={1: torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0.5], [0, 0, -0.5, 0]], dtype=torch.float32)},
    )
    with knob.attach(a_qwen, dose_s=0.4):
        _ = qwen(prompts[0])
        assert qwen.last_tuple_tail == ("kv_stub", 42)
    assert len(qwen.model.layers[1]._forward_hooks) == 0


def test_extract_tangent_basis_orthogonality_and_skew_symmetry() -> None:
    enforce_precision_and_seed(20261117)
    model = _QwenStyleLM(d_model=32, n_layers=2)
    adapter = starbody.HuggingFaceAdapter(model)

    gen = torch.Generator().manual_seed(23)
    pos = [torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) + 0.5 for _ in range(5)]
    neg = [torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) - 0.5 for _ in range(5)]

    u_map, a_map = adapter.extract_tangent_basis(pos, neg, layer_indices=[0, 1], rank_k=4)
    for l_idx in (0, 1):
        u_t = u_map[l_idx]
        a_t = a_map[l_idx]
        assert u_t.dtype == torch.float32 and u_t.shape == (32, 4)
        assert a_t.dtype == torch.float32 and a_t.shape == (4, 4)

        u_64 = u_t.to(torch.float64)
        a_64 = a_t.to(torch.float64)
        ortho_err = float(torch.linalg.norm(u_64.T @ u_64 - torch.eye(4, dtype=torch.float64)).item())
        skew_err = float(torch.linalg.norm(a_64 + a_64.T).item())
        assert ortho_err < 1e-5, f"Layer {l_idx} orthogonality error {ortho_err}"
        assert skew_err < 1e-6, f"Layer {l_idx} skew-symmetry error {skew_err}"
        assert float(torch.linalg.norm(a_64).item()) > 1e-3


def test_geodesic_norm_preservation_all_knobs_and_cleanup() -> None:
    enforce_precision_and_seed(20261117)
    model = _QwenStyleLM(d_model=32, n_layers=2).eval()
    model.requires_grad_(False)
    adapter = starbody.HuggingFaceAdapter(model)

    gen = torch.Generator().manual_seed(31)
    pos = [torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) + 0.3 for _ in range(4)]
    neg = [torch.randn(1, 5, 32, generator=gen, dtype=torch.float32) - 0.3 for _ in range(4)]
    u_map, a_map = adapter.extract_tangent_basis(pos, neg, layer_indices=[0], rank_k=4)

    probe = torch.randn(2, 5, 32, generator=gen, dtype=torch.float32)
    base_act = adapter.capture_activations([probe], [0])[0]
    with torch.no_grad():
        base_out = model(probe)

    # 1. GeodesicSOKKnob must preserve L2 norm inside layer 0 to < 1e-5 and remove hooks cleanly on exception
    knob = starbody.GeodesicSOKKnob({0: u_map[0]}, {0: a_map[0]})
    with pytest.raises(RuntimeError, match="forced_exit"):
        with knob.attach(adapter, dose_s=0.85):
            assert len(model.model.layers[0]._forward_hooks) == 1
            steered_act = adapter.capture_activations([probe], [0])[0]
            # Wait: verify norm of every token state before & after rotation matches within 1e-5
            n_clean = torch.linalg.norm(base_act.to(torch.float64), dim=-1)
            n_steered = torch.linalg.norm(steered_act.to(torch.float64), dim=-1)
            max_rel_diff = float(torch.max(torch.abs(n_steered - n_clean) / n_clean.clamp(min=1e-12)).item())
            assert max_rel_diff < 1e-5, f"Norm violated under SO(k) rotation: rel_diff={max_rel_diff}"
            assert float(torch.linalg.norm((steered_act - base_act).to(torch.float64)).item()) > 1e-3
            raise RuntimeError("forced_exit")

    assert len(model.model.layers[0]._forward_hooks) == 0
    with torch.no_grad():
        after_out = model(probe)
    assert torch.allclose(base_out, after_out, atol=1e-6)

    # 2. LinearPatchKnob & QuantizationStepKnob non-destructive restoration
    lin = starbody.LinearPatchKnob({1: u_map[0][:, 0]})
    with lin.attach(adapter, dose_s=0.5):
        assert len(model.model.layers[1]._forward_hooks) == 1
    assert len(model.model.layers[1]._forward_hooks) == 0

    orig_w = model.model.layers[1].fc1.weight.detach().clone()
    qknob = starbody.QuantizationStepKnob(target_layers=[1])
    with qknob.attach(adapter, dose_s=0.05):
        assert not torch.equal(model.model.layers[1].fc1.weight, orig_w)
    assert torch.equal(model.model.layers[1].fc1.weight, orig_w)

    # 3. LoRAAlphaKnob scaling & restoration + AgentProxyAdapter turn evaluation
    lora_root = _DummyLoRAModule(d_model=16)
    lora_root.model = nn.Module()
    lora_root.model.layers = nn.ModuleList([_TensorLayer(16)])
    lora_adapter = starbody.HuggingFaceAdapter(lora_root)
    with starbody.LoRAAlphaKnob(lora_root).attach(lora_adapter, dose_s=2.5):
        assert lora_root.scaling["default"] == pytest.approx(2.5)
    assert lora_root.scaling["default"] == pytest.approx(1.0)

    proxy = starbody.AgentProxyAdapter(state_dim=32)
    metrics = proxy.evaluate_turns([
        {"agent": "Planner", "content": "Propose safe geodesic intervention"},
        {"agent": "Coder", "content": "Implement matrix_exp hook cleanly"},
        {"agent": "Auditor", "content": "Verify orthogonality in float64"},
        {"agent": "Synthesizer", "content": "Compile signed starbody cartridge"},
    ])
    assert metrics["node_embeddings_shape"] == [4, 32]
    assert metrics["fiedler_lambda2"] > 0.0


def test_cartridge_compile_verify_apply_and_tamper_detection(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _QwenStyleLM(d_model=32, n_layers=2).eval()
    gen = torch.Generator().manual_seed(44)
    pairs = [
        (torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) + 0.8, torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) - 0.8)
        for _ in range(4)
    ]
    canaries = [torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) for _ in range(2)]

    conn = starbody.connect(
        model,
        intervention="geodesic_so_k",
        target_pairs=pairs,
        canaries=canaries,
        serving_dtype="bfloat16",
        layer_fraction=1.0,
    )
    cal = conn.calibrate()
    assert cal["d_model"] == 32
    assert cal["max_ortho_err"] < 1e-5
    assert cal["max_skew_err"] < 1e-6

    cart_path = tmp_path / "test_qwen.starbody"
    # Optional custom HMAC Ed25519-style signer test
    secret = b"starbody-test-signing-key"
    signer = lambda msg: hmac.new(secret, msg, hashlib.sha256).digest()
    verifier = lambda msg, sig: hmac.compare_digest(hmac.new(secret, msg, hashlib.sha256).digest(), sig)

    conn.compile(cart_path, signing_key=signer)
    assert cart_path.exists()

    verified = starbody.verify_cartridge(cart_path, public_key=verifier, require_signature=True)
    assert verified.signature_present is True
    assert verified.signature_valid is True
    assert verified.manifest["serialization_format"] == "safetensors"
    assert len(verified.ledger_head_sha256) == 64

    # Apply onto clean model and verify detach cleanup
    probe = canaries[0]
    with torch.no_grad():
        baseline_out = model(probe)
    handle = starbody.apply_cartridge(model, cart_path, dose_s=0.25, guard=False, verify_signature=True, public_key=verifier)
    assert len(model.model.layers[0]._forward_hooks) == 1
    with torch.no_grad():
        steered_out = model(probe)
    assert not torch.allclose(baseline_out, steered_out, atol=1e-5)
    model.starbody_detach()
    assert handle is not None and len(model.model.layers[0]._forward_hooks) == 0

    # Dimension mismatch fail-closed test
    wrong_dim_model = _QwenStyleLM(d_model=16, n_layers=2)
    with pytest.raises(starbody.StarbodyIntegrityError, match="d_model mismatch"):
        starbody.apply_cartridge(wrong_dim_model, cart_path)

    # 1-byte tamper test inside tensors.safetensors MUST raise StarbodyIntegrityError
    tampered_path = tmp_path / "tampered.starbody"
    with zipfile.ZipFile(cart_path, "r") as zin, zipfile.ZipFile(tampered_path, "w") as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            if item.filename == "tensors.safetensors":
                b_arr = bytearray(raw)
                b_arr[-10] ^= 0xFF
                raw = bytes(b_arr)
            zout.writestr(item.filename, raw)

    with pytest.raises(starbody.StarbodyIntegrityError, match="SHA-256 mismatch"):
        starbody.verify_cartridge(tampered_path)


def test_cartridge_guard_live_dose_attenuation(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    model = _QwenStyleLM(d_model=32, n_layers=2).eval()
    gen = torch.Generator().manual_seed(55)
    clean_input = torch.zeros(1, 4, 32, dtype=torch.float32)
    pairs = [
        (torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) * 0.1 + 0.5, torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) * 0.1 - 0.5)
        for _ in range(4)
    ]

    conn = starbody.connect(model, target_pairs=pairs, canaries=[clean_input], layer_fraction=1.0)
    conn.calibrate()
    cart_path = tmp_path / "guarded.starbody"
    conn.compile(cart_path, max_e_perp_budget=0.01)

    handle = starbody.apply_cartridge(model, cart_path, dose_s=0.40, guard=True)
    try:
        # Clean pass stays at full dose 0.40
        with torch.no_grad():
            _ = model(clean_input)
        assert model._starbody_guard_state.attenuation_count == 0
        assert model._starbody_guard_state.active_dose_s == pytest.approx(0.40)

        # Synthetic off-manifold drift perturbation orthogonal to U_parallel triggers 50% attenuation -> 0.20
        u0_64 = handle.verified.tensors["layer_0.u_parallel"].to(torch.float64)
        rand_orth = torch.randn(32, generator=gen, dtype=torch.float64)
        rand_orth = rand_orth - u0_64 @ (u0_64.T @ rand_orth)
        rand_orth = (rand_orth / torch.norm(rand_orth).clamp(min=1e-12)).to(torch.float32)
        drifted_input = clean_input + 6.0 * rand_orth.view(1, 1, 32)

        with pytest.warns(RuntimeWarning, match="attenuating dose 50%"):
            with torch.no_grad():
                _ = model(drifted_input)

        assert model._starbody_guard_state.attenuation_count >= 1
        assert model._starbody_guard_state.last_e_perp > 0.01
        # Two target layers can each trigger on the drifted batch, cutting by 0.5 per exceeded layer
        assert model._starbody_guard_state.active_dose_s <= 0.20 + 1e-6
    finally:
        model.starbody_detach()


@pytest.mark.skipif(srv is None or TestClient is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_worker_ws_command_routing_and_cuda_vram(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False

    # Verify CUDA RTX 3050 float32 onboarding stays far under 3.0 GB if CUDA is available
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _QwenStyleLM(d_model=64, n_layers=2).to(dev)
    gen = torch.Generator().manual_seed(99)
    pairs = [
        (torch.randn(1, 4, 64, generator=gen, dtype=torch.float32) + 0.5, torch.randn(1, 4, 64, generator=gen, dtype=torch.float32) - 0.5)
        for _ in range(4)
    ]
    canaries = [torch.randn(1, 4, 64, generator=gen, dtype=torch.float32) for _ in range(2)]
    conn = starbody.connect(
        model,
        target_pairs=pairs,
        canaries=canaries,
        layer_fraction=1.0,
        device=dev,
        model_name="_QwenStyleLM-Test",
    )
    cal = conn.calibrate()
    if torch.cuda.is_available():
        assert cal["vram_peak_mb"] <= 3072.0

    with TestClient(srv.app) as client:
        with client.websocket_connect("/ws/telemetry") as ws_tel:
            _ = ws_tel.receive_json()

            with client.websocket_connect("/ws/worker") as ws_worker:
                ws_worker.send_json({
                    "type": "register",
                    "model_name": "_QwenStyleLM-Test",
                    "device": str(dev),
                    "d_model": 64,
                    "hook_path": "model.layers",
                    "intervention": "geodesic_so_k",
                    "calibration": cal,
                })
                reg_bcast = ws_tel.receive_json()
                assert reg_bcast["worker"]["connected"] is True
                assert reg_bcast["worker"]["model_name"] == "_QwenStyleLM-Test"

                # Issue TEST_PROMPT command from REST thread while worker processes WS frame
                resp_box: Dict[str, Any] = {}

                def _call_prompt() -> None:
                    r = client.post(
                        "/api/worker/command",
                        json={"command": "TEST_PROMPT", "params": {"prompt": "Verify alignment.", "dose_s": 0.1}},
                    )
                    resp_box["status_code"] = r.status_code
                    resp_box["json"] = r.json()

                t = threading.Thread(target=_call_prompt, daemon=True)
                t.start()

                cmd_frame = ws_worker.receive_json()
                assert cmd_frame["command"] == "TEST_PROMPT"
                prompt_res = conn.test_prompt(cmd_frame["params"]["prompt"], dose_s=cmd_frame["params"]["dose_s"])
                ws_worker.send_json({
                    "type": "command_result",
                    "request_id": cmd_frame["request_id"],
                    "command": "TEST_PROMPT",
                    "status": "ok",
                    "payload": prompt_res,
                })
                t.join(timeout=5.0)
                assert resp_box.get("status_code") == 200
                assert resp_box["json"]["mode"] == "remote_worker"
                assert "Steered" in resp_box["json"]["result"]["steered_output"]

                # Issue COMPILE_CARTRIDGE over WS worker & verify download endpoint
                out_cart = tmp_path / "ws_compiled.starbody"
                resp_box.clear()

                def _call_compile() -> None:
                    r = client.post(
                        "/api/worker/command",
                        json={"command": "COMPILE_CARTRIDGE", "params": {"output_path": str(out_cart)}},
                    )
                    resp_box["status_code"] = r.status_code
                    resp_box["json"] = r.json()

                t2 = threading.Thread(target=_call_compile, daemon=True)
                t2.start()

                cmd2 = ws_worker.receive_json()
                assert cmd2["command"] == "COMPILE_CARTRIDGE"
                compiled_p = conn.compile(cmd2["params"]["output_path"])
                ws_worker.send_json({
                    "type": "command_result",
                    "request_id": cmd2["request_id"],
                    "command": "COMPILE_CARTRIDGE",
                    "status": "ok",
                    "payload": {
                        "cartridge_path": str(compiled_p.resolve()),
                        "filename": compiled_p.name,
                        "size_bytes": compiled_p.stat().st_size,
                    },
                })
                t2.join(timeout=5.0)
                assert resp_box.get("status_code") == 200

                dl_resp = client.get("/api/worker/download_cartridge")
                assert dl_resp.status_code == 200
                dl_file = tmp_path / "downloaded.starbody"
                dl_file.write_bytes(dl_resp.content)
                verified_dl = starbody.verify_cartridge(dl_file)
                assert verified_dl.manifest["model_name"] == "_QwenStyleLM-Test"


@pytest.mark.skipif(srv is None or TestClient is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_v33_workbench_html_and_worker_commands(tmp_path: Path) -> None:
    uploaded_path: Path | None = None
    with TestClient(srv.app) as client:
        try:
            html_resp = client.get("/")
            assert html_resp.status_code == 200
            html = html_resp.text
            for dom_id in (
                "view-connect",
                "view-calibrate",
                "view-dose-diff",
                "view-undo-clock",
                "view-ship-audit",
                "heroMode3dBtn",
                "heroModeSplitBtn",
                "heroModeOutputBtn",
                "gateTf32",
                "gateSham",
                "gateOrtho",
                "gateServing",
                "snapCertifiedBtn",
                "canaryHealthMatrix",
                "undoClockMultiplier",
                "compileCartridgeBtn",
                "copyLoaderBtn",
                "copyCiYamlBtn",
            ):
                assert dom_id in html, f"Missing v3.3 workbench token: {dom_id}"
            assert "https://" not in html

            sb_resp = client.post("/api/worker/command", json={"command": "LAUNCH_SANDBOX", "params": {}})
            assert sb_resp.status_code == 200
            sb_res = sb_resp.json()["result"]
            assert sb_res["sandbox_active"] is True
            assert sb_res["last_calibration"]["gates"]["tf32_locked"] is True

            cfg_resp = client.post(
                "/api/worker/command",
                json={
                    "command": "UPDATE_CONFIG",
                    "params": {
                        "target_pairs": [["Precision prompt A", "Noisy prompt B"], ["Exact answer C", "Vague answer D"]],
                        "canaries": ["2 + 2 =", "Water freezes at"],
                        "intervention": "geodesic_so_k",
                        "serving_dtype": "bfloat16",
                    },
                },
            )
            assert cfg_resp.status_code == 200
            assert cfg_resp.json()["result"]["canaries"] == ["2 + 2 =", "Water freezes at"]

            cal_resp = client.post("/api/worker/command", json={"command": "RUN_CALIBRATION", "params": {}})
            assert cal_resp.status_code == 200
            gates = cal_resp.json()["result"]["gates"]
            assert gates["tf32_locked"] is True
            assert gates["sham_floor_ok"] is True
            assert gates["ortho_ok"] is True
            assert gates["serving_floor_ok"] is True

            prm_resp = client.post(
                "/api/worker/command",
                json={"command": "TEST_PROMPT", "params": {"prompt": "2 + 2 =", "dose_s": 0.1}},
            )
            assert prm_resp.status_code == 200
            assert "Steered" in prm_resp.json()["result"]["steered_output"]

            cart_out = tmp_path / "workbench_test.starbody"
            cmp_resp = client.post(
                "/api/worker/command",
                json={"command": "COMPILE_CARTRIDGE", "params": {"output_path": str(cart_out)}},
            )
            assert cmp_resp.status_code == 200
            assert cart_out.exists()

            raw_bytes = cart_out.read_bytes()
            up_resp = client.post(
                "/api/cartridge/upload",
                json={"filename": "workbench_test.starbody", "content_b64": base64.b64encode(raw_bytes).decode("ascii")},
            )
            assert up_resp.status_code == 200
            up_data = up_resp.json()
            assert up_data["status"] == "verified"
            uploaded_path = Path(up_data["cartridge"]["cartridge_path"])
            assert uploaded_path.exists()

            # Tamper 1 byte inside tensors.safetensors -> fail-closed 400
            tampered_zip = tmp_path / "tampered.starbody"
            with zipfile.ZipFile(cart_out, "r") as zin, zipfile.ZipFile(tampered_zip, "w") as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "tensors.safetensors":
                        ba = bytearray(data)
                        ba[-1] ^= 0xFF
                        data = bytes(ba)
                    zout.writestr(item, data)
            bad_resp = client.post(
                "/api/cartridge/upload",
                json={"filename": "tampered.starbody", "content_b64": base64.b64encode(tampered_zip.read_bytes()).decode("ascii")},
            )
            assert bad_resp.status_code == 400
        finally:
            srv._LOCAL_FALLBACK_CONN = None
            if uploaded_path is not None and uploaded_path.exists():
                uploaded_path.unlink(missing_ok=True)


def test_ed25519_keygen_sign_verify_and_tamper_rejection(tmp_path: Path) -> None:
    enforce_precision_and_seed(20261117)
    priv_p, pub_p = starbody.generate_ed25519_keypair(tmp_path / "key.pem")
    _, wrong_pub = starbody.generate_ed25519_keypair(tmp_path / "wrong.pem")
    assert priv_p.exists() and pub_p.exists()

    q, _ = torch.linalg.qr(torch.randn(16, 4, dtype=torch.float64))
    u = q.to(torch.float32)
    a = torch.zeros((4, 4), dtype=torch.float32)
    a[0, 1], a[1, 0] = 1.0, -1.0

    manifest = {
        "format_version": "starbody.v1",
        "model_name": "Ed25519Unit",
        "hook_path": "model.layers",
        "d_model": 16,
        "layer_indices": [0],
        "rank_k": 4,
        "intervention": "geodesic_so_k",
        "recommended_dose_s": 0.1,
        "s_cert_ceiling": 0.4,
    }
    tensors = {"layer_0.u_parallel": u, "layer_0.a_skew": a}
    receipt = {"protocol": "SMP v3.3", "ok": True}

    signed_cart = starbody.compile_cartridge(
        tmp_path / "signed.starbody",
        manifest_dict=manifest,
        tensors_dict=tensors,
        receipt_dict=receipt,
        signing_key=priv_p,
    )
    ver = starbody.verify_cartridge(signed_cart, public_key=pub_p, require_signature=True)
    assert ver.signature_present is True
    assert ver.signature_valid is True

    with pytest.raises(starbody.StarbodyIntegrityError):
        starbody.verify_cartridge(signed_cart, public_key=wrong_pub, require_signature=True)

    unsigned_cart = starbody.compile_cartridge(
        tmp_path / "unsigned.starbody",
        manifest_dict=manifest,
        tensors_dict=tensors,
        receipt_dict=receipt,
        signing_key=None,
    )
    with pytest.raises(starbody.StarbodyIntegrityError):
        starbody.verify_cartridge(unsigned_cart, public_key=pub_p, require_signature=True)


def test_geodesic_per_row_multitenant_batch_and_position_modes() -> None:
    enforce_precision_and_seed(20261117)
    model = _QwenStyleLM(d_model=32, n_layers=1).eval()
    model.requires_grad_(False)
    adapter = starbody.HuggingFaceAdapter(model)

    gen = torch.Generator().manual_seed(303)
    pairs = [
        (torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) + 0.5, torch.randn(1, 6, 32, generator=gen, dtype=torch.float32) - 0.5)
        for _ in range(4)
    ]
    u_map, a_map = adapter.extract_tangent_basis([p[0] for p in pairs], [p[1] for p in pairs], layer_indices=[0], rank_k=4)
    knob = starbody.GeodesicSOKKnob(u_map, a_map)

    x_batch = torch.randn(3, 6, 32, generator=gen, dtype=torch.float32)
    with torch.no_grad():
        base_out = model(x_batch)

    # Multi-tenant batch: row 0 unsteered (dose=0.0), row 1 positive, row 2 negative
    with knob.attach(adapter, dose_s=[0.0, 0.35, -0.25]):
        with torch.no_grad():
            steered_out = model(x_batch)

    assert torch.allclose(steered_out[0], base_out[0], atol=1e-6)
    assert float(torch.norm(steered_out[1] - base_out[1]).item()) > 1e-3
    assert float(torch.norm(steered_out[2] - base_out[2]).item()) > 1e-3

    # Position mode: last_token keeps prefix identical and steers only final position
    with knob.attach(adapter, dose_s=0.40, position_mode="last_token"):
        with torch.no_grad():
            lt_out = model(x_batch)
    assert torch.allclose(lt_out[:, :-1, :], base_out[:, :-1, :], atol=1e-6)
    assert float(torch.norm(lt_out[:, -1:, :] - base_out[:, -1:, :]).item()) > 1e-3

    # Position mode: prefill_only vs decode_only
    x_decode = x_batch[:, :1, :]
    with torch.no_grad():
        base_dec = model(x_decode)
    with knob.attach(adapter, dose_s=0.40, position_mode="prefill_only"):
        with torch.no_grad():
            pref_full = model(x_batch)
            pref_dec = model(x_decode)
    assert float(torch.norm(pref_full - base_out).item()) > 1e-3
    assert torch.allclose(pref_dec, base_dec, atol=1e-6)

    with knob.attach(adapter, dose_s=0.40, position_mode="decode_only"):
        with torch.no_grad():
            dec_full = model(x_batch)
            dec_step = model(x_decode)
    assert torch.allclose(dec_full, base_out, atol=1e-6)
    assert float(torch.norm(dec_step - base_dec).item()) > 1e-3
    assert len(model.model.layers[0]._forward_hooks) == 0


def test_lora_alpha_connect_and_vram_preflight_guard() -> None:
    enforce_precision_and_seed(20261117)

    class _LoRALayer(nn.Module):
        def __init__(self, d_model: int = 32) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_model)
            self.scaling = 1.0

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str, int]:
            return (x + self.scaling * 0.2 * torch.tanh(self.fc1(x)), "kv", 1)

    class _LoRALM(nn.Module):
        def __init__(self, d_model: int = 32) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_LoRALayer(d_model) for _ in range(2)])
            self.lm_head = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for blk in self.model.layers:
                h = blk(h)[0]
            return self.lm_head(h)

    model = _LoRALM(d_model=32).eval()
    gen = torch.Generator().manual_seed(81)
    pairs = [
        (torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) + 0.4, torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) - 0.4)
        for _ in range(4)
    ]
    canaries = [torch.randn(1, 4, 32, generator=gen, dtype=torch.float32) for _ in range(2)]

    conn = starbody.connect(model, intervention="lora_alpha", target_pairs=pairs, canaries=canaries, layer_fraction=1.0)
    cal = conn.calibrate()
    assert cal["gates"]["ortho_ok"] is True
    lad = conn.sweep_ladder([0.0, 0.5, 1.0])
    assert len(lad["rungs"]) == 3
    assert model.model.layers[0].scaling == 1.0

    # Verify RTX 3050 FP32 VRAM pre-check rejects oversized configurations fail-closed
    if srv is not None and TestClient is not None:
        with TestClient(srv.app) as client:
            avail_resp = client.get("/api/models/available")
            assert avail_resp.status_code == 200
            assert any(m["model_id"] == "LocalSandboxLM-FP32" for m in avail_resp.json()["models"])

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M"
            if hf_cache.exists():
                rej_resp = client.post(
                    "/api/worker/load_model",
                    json={"model_id": "HuggingFaceTB/SmolLM2-135M", "max_vram_mb": 1.0, "local_files_only": True},
                )
                assert rej_resp.status_code == 400
                assert "exceeding RTX 3050 budget" in rej_resp.json()["detail"]


@pytest.mark.skipif(srv is None or TestClient is None, reason="starbody_studio_server not installed in standalone starbody-core")
def test_real_smollm2_rtx3050_load_prompt_latency_and_signed_cartridge(tmp_path: Path) -> None:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M"
    if not hf_cache.exists():
        pytest.skip("HuggingFaceTB/SmolLM2-135M not cached locally")

    with TestClient(srv.app) as client:
        try:
            load_resp = client.post(
                "/api/worker/load_model",
                json={
                    "model_id": "HuggingFaceTB/SmolLM2-135M",
                    "intervention": "geodesic_so_k",
                    "local_files_only": True,
                },
            )
            assert load_resp.status_code == 200, load_resp.text
            load_data = load_resp.json()
            m_info = load_data["model"]
            assert m_info["model_name"] == "HuggingFaceTB/SmolLM2-135M"
            assert m_info["num_layers"] == 30
            assert m_info["d_model"] == 576
            cal = m_info["last_calibration"]
            assert cal["gates"]["tf32_locked"] is True
            assert cal["gates"]["ortho_ok"] is True
            if torch.cuda.is_available():
                assert cal["vram_peak_mb"] <= 3072.0

            prm_resp = client.post(
                "/api/worker/command",
                json={"command": "TEST_PROMPT", "params": {"prompt": "The capital of France is", "dose_s": 0.25}},
            )
            assert prm_resp.status_code == 200
            prm = prm_resp.json()["result"]
            assert prm["baseline_output"] and not prm["baseline_output"].startswith("Baseline logit indices")
            assert prm["steered_output"] and not prm["steered_output"].startswith("Steered (s=")
            assert len(prm["baseline_output"].split()) >= 4
            assert len(prm["steered_output"].split()) >= 4
            assert prm["baseline_ms_per_tok"] > 0.0
            assert prm["steered_ms_per_tok"] > 0.0
            assert prm["tokens_generated"] >= 1

            cart_out = tmp_path / "smollm2.starbody"
            cmp_resp = client.post(
                "/api/worker/command",
                json={"command": "COMPILE_CARTRIDGE", "params": {"output_path": str(cart_out)}},
            )
            assert cmp_resp.status_code == 200
            cmp_res = cmp_resp.json()["result"]
            assert cmp_res["signature_present"] is True
            ver = starbody.verify_cartridge(
                cart_out,
                public_key=cmp_res["public_key_path"],
                require_signature=True,
            )
            assert ver.manifest["model_name"] == "HuggingFaceTB/SmolLM2-135M"
            assert ver.signature_valid is True
        finally:
            srv._LOCAL_FALLBACK_CONN = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def test_cli_keygen_subcommand(tmp_path: Path) -> None:
    rc = starbody_cli.main(["keygen", "--out-dir", str(tmp_path / "keys")])
    assert rc == 0
    priv_p = tmp_path / "keys" / "starbody_ed25519.pem"
    pub_p = tmp_path / "keys" / "starbody_ed25519.pub.pem"
    assert priv_p.exists() and pub_p.exists()
    _ = starbody.load_ed25519_private_key(priv_p)
    _ = starbody.load_ed25519_public_key(pub_p)

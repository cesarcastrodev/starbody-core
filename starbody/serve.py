"""
OpenAI-Compatible Multi-Tenant Steering Serving Proxy (SMP v3.3 Sprint 2).

Serves one shared float32 GPU/CPU backbone simultaneously across multiple verified
`.starbody` cartridges with per-request `model:cartridge` aliases, `X-Starbody-*` headers,
SSE streaming, fail-closed Ed25519 verification, and single-batch multi-tenant routing
where zero-dose rows stay bit-exact with baseline while non-zero rows rotate on SO(k).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

import torch
import torch.nn as nn

try:
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = Any  # type: ignore[misc,assignment]

from smp_core import enforce_precision_and_seed
from starbody.adapters import HuggingFaceAdapter
from starbody.cartridge import (
    GuardRuntimeState,
    StarbodyIntegrityError,
    VerifiedCartridge,
    _extract_layer_tensors_from_dict,
    build_normalized_guard_fn,
    verify_cartridge,
)
from starbody.connect import estimate_fp32_vram_mb
from starbody.interventions import AttachedController, GeodesicSOKKnob


def _peak_vram_mb(dev: Optional[torch.device] = None) -> float:
    if torch.cuda.is_available() and torch.cuda.is_initialized() and (dev is None or dev.type == "cuda"):
        return round(float(torch.cuda.max_memory_allocated(dev if dev is not None else 0) / (1024 ** 2)), 2)
    return 0.0


def _extract_primary(raw_out: Any) -> torch.Tensor:
    if hasattr(raw_out, "logits") and isinstance(raw_out.logits, torch.Tensor):
        return raw_out.logits
    if isinstance(raw_out, (tuple, list)) and isinstance(raw_out[0], torch.Tensor):
        return raw_out[0]
    if isinstance(raw_out, torch.Tensor):
        return raw_out
    raise TypeError(f"Unsupported output type {type(raw_out).__name__}")


def _format_chat_prompt(messages: Sequence[Mapping[str, Any]], tokenizer: Optional[Any]) -> str:
    cleaned = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in cleaned]
    lines.append("Assistant:")
    return "\n".join(lines)


class StarbodyServingRuntime:
    """Shared strict-FP32 model host supporting hot-loaded cartridges and batched multi-tenancy."""

    def __init__(
        self,
        model: Union[nn.Module, str],
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        model_name: Optional[str] = None,
        local_files_only: bool = False,
        max_vram_mb: float = 3072.0,
    ) -> None:
        enforce_precision_and_seed(20261117)
        if isinstance(model, str):
            is_local = bool(local_files_only or Path(model).exists())
            est = estimate_fp32_vram_mb(model, local_files_only=is_local, budget_mb=max_vram_mb)
            if not est["fits_rtx3050_budget"]:
                raise ValueError(
                    f"Model '{model}' estimated at {est['est_fp32_mb']} MB in strict FP32, "
                    f"exceeding RTX 3050 budget ({max_vram_mb:.0f} MB)."
                )
            from transformers import AutoModelForCausalLM, AutoTokenizer

            target_dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=is_local)
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            loaded = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=is_local,
            )
            loaded.to(target_dev)
            nn_model = loaded
            resolved_name = model_name or model
        else:
            nn_model = model
            existing_dev = HuggingFaceAdapter._infer_device(nn_model)
            target_dev = device or existing_dev
            nn_model.to(dtype=torch.float32, device=target_dev)
            resolved_name = model_name or getattr(getattr(model, "config", None), "_name_or_path", None) or type(model).__name__

        nn_model.eval()
        nn_model.requires_grad_(False)
        self.model = nn_model
        self.model_name = str(resolved_name)
        self.adapter = HuggingFaceAdapter(nn_model, tokenizer=tokenizer, device=target_dev)
        self.cartridges: Dict[str, VerifiedCartridge] = {}
        self.cartridge_paths: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def load_cartridge(
        self,
        name: str,
        cartridge_path: Union[str, Path],
        public_key: Any = None,
        require_signature: bool = False,
    ) -> VerifiedCartridge:
        ver = verify_cartridge(cartridge_path, public_key=public_key, require_signature=require_signature)
        expected_d = ver.manifest.get("d_model")
        if expected_d is not None and int(expected_d) != self.adapter.d_model:
            raise StarbodyIntegrityError(
                f"Model d_model mismatch: cartridge '{name}' expects {expected_d}, model has {self.adapter.d_model}"
            )
        expected_hook = ver.manifest.get("hook_path")
        if expected_hook is not None and str(expected_hook) != self.adapter.hook_path_prefix():
            raise StarbodyIntegrityError(
                f"Hook path mismatch: cartridge '{name}' expects '{expected_hook}', model uses '{self.adapter.hook_path_prefix()}'"
            )
        self.cartridges[name] = ver
        self.cartridge_paths[name] = str(Path(cartridge_path).resolve())
        return ver

    def parse_steering_spec(
        self,
        model_field: Optional[str],
        starbody_body: Optional[Mapping[str, Any]],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        sb = dict(starbody_body or {})
        cartridge_name = sb.get("cartridge") or headers.get("x-starbody-cartridge")
        if not cartridge_name and model_field and ":" in model_field:
            _, candidate = model_field.split(":", 1)
            if candidate in self.cartridges:
                cartridge_name = candidate

        if not cartridge_name:
            return {"cartridge": None, "dose_s": 0.0, "position_mode": "all", "guard": True, "clamped": False}

        if cartridge_name not in self.cartridges:
            raise KeyError(f"Unknown mounted cartridge '{cartridge_name}'. Available: {sorted(self.cartridges.keys())}")

        ver = self.cartridges[cartridge_name]
        ceiling = float(ver.manifest.get("s_cert_ceiling", 10.0))
        rec_dose = float(ver.manifest.get("recommended_dose_s", min(0.1, ceiling)))
        raw_dose = sb.get("dose_s")
        if raw_dose is None and headers.get("x-starbody-dose") is not None:
            raw_dose = float(headers["x-starbody-dose"])
        requested_dose = rec_dose if raw_dose is None else float(raw_dose)

        strict_bounds = bool(sb.get("strict_bounds", False)) or (
            str(headers.get("x-starbody-strict-bounds", "")).lower() in ("1", "true", "yes")
        )
        if strict_bounds and abs(requested_dose) > ceiling + 1e-12:
            raise ValueError(
                f"Requested dose_s={requested_dose} exceeds cartridge '{cartridge_name}' certified ceiling s_cert={ceiling}."
            )

        clamped_dose = max(-ceiling, min(ceiling, requested_dose))
        pos_mode = str(sb.get("position_mode") or headers.get("x-starbody-position-mode") or "all").lower().strip()
        guard_on = bool(sb.get("guard", True))

        return {
            "cartridge": cartridge_name,
            "requested_dose_s": requested_dose,
            "dose_s": clamped_dose,
            "clamped": abs(clamped_dose - requested_dose) > 1e-12,
            "s_cert_ceiling": ceiling,
            "position_mode": pos_mode,
            "guard": guard_on,
        }

    def _build_cartridge_knob(
        self,
        ver: VerifiedCartridge,
        chosen_dose: float,
        guard: bool,
        position_mode: str,
    ) -> Tuple[GeodesicSOKKnob, Optional[GuardRuntimeState]]:
        u_raw, a_raw, base_raw = _extract_layer_tensors_from_dict(ver.tensors)
        u_map = {idx: t.to(device=self.adapter.device, dtype=torch.float32) for idx, t in u_raw.items()}
        a_map = {idx: t.to(device=self.adapter.device, dtype=torch.float32) for idx, t in a_raw.items()}
        baselines_64 = {idx: t.to(device=self.adapter.device, dtype=torch.float64) for idx, t in base_raw.items()}

        guard_fn = None
        guard_state: Optional[GuardRuntimeState] = None
        if guard:
            guard_fn, guard_state = build_normalized_guard_fn(ver, baselines_64, chosen_dose, emit_warnings=False)

        knob = GeodesicSOKKnob(
            u_parallel_by_layer=u_map,
            a_skew_by_layer=a_map,
            guard_fn=guard_fn,
            position_mode=position_mode,
            normalize_by_layers=bool(ver.manifest.get("normalize_by_layers", False)),
        )
        return knob, guard_state

    def _sync(self) -> None:
        if self.adapter.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.adapter.device)

    @torch.no_grad()
    def generate_single(
        self,
        prompt_text: str,
        max_tokens: int = 48,
        temperature: float = 0.0,
        steering: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = steering or {"cartridge": None, "dose_s": 0.0, "position_mode": "all", "guard": True}
        cname = spec.get("cartridge")
        dose_val = float(spec.get("dose_s", 0.0))
        pos_mode = str(spec.get("position_mode", "all"))

        ctx = None
        guard_state: Optional[GuardRuntimeState] = None
        if cname and abs(dose_val) > 1e-12:
            knob, guard_state = self._build_cartridge_knob(
                self.cartridges[cname],
                chosen_dose=dose_val,
                guard=bool(spec.get("guard", True)),
                position_mode=pos_mode,
            )
            ctx = knob.attach(self.adapter, dose_s=dose_val, position_mode=pos_mode)

        tok = self.adapter.tokenizer
        self._sync()
        t0 = time.perf_counter()

        try:
            if ctx is not None:
                ctx.__enter__()

            if tok is not None and hasattr(self.model, "generate"):
                enc = self.adapter._encode_string_prompt(prompt_text)
                if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                    tok.pad_token_id = tok.eos_token_id
                in_len = int(enc["input_ids"].shape[1]) if isinstance(enc, dict) else 1
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max(1, min(256, int(max_tokens))),
                    "do_sample": bool(temperature > 0.0),
                }
                if temperature > 0.0:
                    gen_kwargs["temperature"] = float(temperature)
                if getattr(tok, "pad_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = int(tok.pad_token_id)
                out_ids = self.model.generate(**enc, **gen_kwargs)
                self._sync()
                elapsed = max(1e-6, time.perf_counter() - t0)
                comp_toks = max(1, int(out_ids.shape[1]) - in_len)
                text = tok.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()
            else:
                raw_out = self.adapter.forward_item(prompt_text)
                self._sync()
                elapsed = max(1e-6, time.perf_counter() - t0)
                t_out = _extract_primary(raw_out).detach().to(torch.float32)
                in_len = int(t_out.shape[1]) if t_out.ndim >= 2 else 1
                comp_toks = 1
                top_vals = t_out[0, -1].topk(min(4, t_out.shape[-1]))
                l2 = float(torch.linalg.norm(t_out.to(torch.float64)).item())
                text = f"starbody[{cname or 'base'} s={dose_val:.3f}] top={top_vals.indices.tolist()} L2={l2:.4f}"
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

        return {
            "text": text,
            "prompt_tokens": in_len,
            "completion_tokens": comp_toks,
            "latency_ms": round(elapsed * 1000.0, 2),
            "ms_per_tok": round((elapsed * 1000.0) / comp_toks, 3),
            "vram_peak_mb": _peak_vram_mb(self.adapter.device),
            "steering": {
                "cartridge": cname,
                "dose_s": dose_val,
                "effective_dose_s": guard_state.active_dose_s if guard_state is not None else dose_val,
                "attenuation_count": guard_state.attenuation_count if guard_state is not None else 0,
                "last_e_perp": guard_state.last_e_perp if guard_state is not None else 0.0,
                "effective_threshold": guard_state.effective_threshold if guard_state is not None else 0.0,
                "position_mode": pos_mode,
            },
        }

    @torch.no_grad()
    def batch_generate_multitenant(
        self,
        prompts: Sequence[str],
        doses: Sequence[float],
        cartridge_name: str,
        max_tokens: int = 24,
        position_mode: str = "all",
        strict_bounds: bool = False,
    ) -> Dict[str, Any]:
        if len(prompts) == 0 or len(prompts) != len(doses):
            raise ValueError("`prompts` and `doses` must be non-empty sequences of equal length.")
        if cartridge_name not in self.cartridges:
            raise KeyError(f"Unknown mounted cartridge '{cartridge_name}'.")

        ver = self.cartridges[cartridge_name]
        ceiling = float(ver.manifest.get("s_cert_ceiling", 10.0))
        clamped_doses: List[float] = []
        for d in doses:
            fd = float(d)
            if strict_bounds and abs(fd) > ceiling + 1e-12:
                raise ValueError(f"Batch dose {fd} exceeds ceiling s_cert={ceiling}.")
            clamped_doses.append(max(-ceiling, min(ceiling, fd)))

        knob, _ = self._build_cartridge_knob(ver, chosen_dose=0.0, guard=False, position_mode=position_mode)
        tok = self.adapter.tokenizer
        self._sync()
        t0 = time.perf_counter()

        results: List[Dict[str, Any]] = []
        if tok is not None and hasattr(self.model, "generate"):
            orig_pad_side = getattr(tok, "padding_side", "right")
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token_id = tok.eos_token_id
            try:
                tok.padding_side = "left"
                enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=256)
                enc_dev = {k: v.to(self.adapter.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
                in_width = int(enc_dev["input_ids"].shape[1])
                gen_kwargs: Dict[str, Any] = {"max_new_tokens": max(1, min(128, int(max_tokens))), "do_sample": False}
                if getattr(tok, "pad_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = int(tok.pad_token_id)
                with knob.attach(self.adapter, dose_s=clamped_doses, position_mode=position_mode):
                    out_ids = self.model.generate(**enc_dev, **gen_kwargs)
                self._sync()
                elapsed = max(1e-6, time.perf_counter() - t0)
                for idx, prompt in enumerate(prompts):
                    text = tok.decode(out_ids[idx, in_width:], skip_special_tokens=True).strip()
                    results.append({
                        "row_index": idx,
                        "prompt": prompt,
                        "dose_s": clamped_doses[idx],
                        "text": text,
                        "completion_tokens": max(1, int(out_ids.shape[1]) - in_width),
                    })
            finally:
                tok.padding_side = orig_pad_side
        else:
            x_rows: List[torch.Tensor] = []
            for p in prompts:
                enc_p = self.adapter._encode_string_prompt(p)
                if not isinstance(enc_p, torch.Tensor):
                    raise TypeError("Expected Tensor inputs for embeddingless batch inference.")
                x_rows.append(enc_p)
            max_seq = max(int(r.shape[1]) for r in x_rows)
            padded = [
                torch.nn.functional.pad(r, (0, 0, 0, max_seq - int(r.shape[1]))) if r.shape[1] < max_seq else r
                for r in x_rows
            ]
            x_batch = torch.cat(padded, dim=0)  # [batch, max_seq, d_model]
            base_out = _extract_primary(self.model(x_batch)).detach().to(torch.float32)
            with knob.attach(self.adapter, dose_s=clamped_doses, position_mode=position_mode):
                steered_out = _extract_primary(self.model(x_batch)).detach().to(torch.float32)
            self._sync()
            elapsed = max(1e-6, time.perf_counter() - t0)
            for idx, prompt in enumerate(prompts):
                delta_l2 = float(torch.linalg.norm((steered_out[idx] - base_out[idx]).to(torch.float64)).item())
                top_idx = steered_out[idx, -1].topk(min(4, steered_out.shape[-1])).indices.tolist()
                results.append({
                    "row_index": idx,
                    "prompt": prompt,
                    "dose_s": clamped_doses[idx],
                    "delta_l2_from_base": delta_l2,
                    "text": f"batch_row[{idx} s={clamped_doses[idx]:.3f}] top={top_idx} delta={delta_l2:.5f}",
                })

        return {
            "model": self.model_name,
            "cartridge": cartridge_name,
            "batch_size": len(prompts),
            "position_mode": position_mode,
            "batch_latency_ms": round(elapsed * 1000.0, 2),
            "vram_peak_mb": _peak_vram_mb(self.adapter.device),
            "results": results,
        }


def create_serving_app(runtime: StarbodyServingRuntime) -> Any:
    """Creates a FastAPI OpenAI-compatible + Starbody multi-tenant HTTP server."""
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    srv_app = FastAPI(title="Starbody OpenAI-Compatible Steering Proxy (SMP v3.3)")

    @srv_app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "model": runtime.model_name,
            "device": str(runtime.adapter.device),
            "d_model": runtime.adapter.d_model,
            "hook_path": runtime.adapter.hook_path_prefix(),
            "cartridges": sorted(runtime.cartridges.keys()),
            "vram_peak_mb": _peak_vram_mb(runtime.adapter.device),
        })

    @srv_app.get("/v1/models")
    def list_models() -> JSONResponse:
        now = int(time.time())
        entries = [{"id": runtime.model_name, "object": "model", "created": now, "owned_by": "starbody"}]
        for cname in sorted(runtime.cartridges.keys()):
            entries.append({"id": f"{runtime.model_name}:{cname}", "object": "model", "created": now, "owned_by": "starbody"})
        return JSONResponse({"object": "list", "data": entries})

    @srv_app.get("/v1/starbody/cartridges")
    def list_cartridges() -> JSONResponse:
        items = []
        for cname, ver in sorted(runtime.cartridges.items()):
            items.append({
                "name": cname,
                "path": runtime.cartridge_paths.get(cname),
                "manifest_sha256": ver.manifest_sha256,
                "d_model": ver.manifest.get("d_model"),
                "layer_indices": ver.manifest.get("layer_indices"),
                "recommended_dose_s": ver.manifest.get("recommended_dose_s"),
                "s_cert_ceiling": ver.manifest.get("s_cert_ceiling"),
                "signature_present": ver.signature_present,
                "signature_valid": ver.signature_valid,
            })
        return JSONResponse({"cartridges": items})

    @srv_app.post("/v1/starbody/cartridges/load")
    async def load_cartridge_endpoint(body: Dict[str, Any] = Body(...)) -> JSONResponse:
        path = str(body.get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="Missing `path`.")
        name = str(body.get("name") or Path(path).stem).strip()
        pub_key = body.get("public_key")
        req_sig = bool(body.get("require_signature", False))
        try:
            async with runtime._lock:
                ver = runtime.load_cartridge(name, path, public_key=pub_key, require_signature=req_sig)
        except StarbodyIntegrityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse({"status": "loaded", "name": name, "manifest_sha256": ver.manifest_sha256})

    def _resolve_steering(req: Request, body: Mapping[str, Any]) -> Dict[str, Any]:
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        try:
            return runtime.parse_steering_spec(
                model_field=body.get("model"),
                starbody_body=body.get("starbody"),
                headers=headers_lower,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @srv_app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: Dict[str, Any] = Body(...)) -> Any:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="Field `messages` must be a non-empty list.")
        steering = _resolve_steering(request, body)
        prompt_text = _format_chat_prompt(messages, runtime.adapter.tokenizer)
        max_tokens = int(body.get("max_tokens", 48))
        temperature = float(body.get("temperature", 0.0))
        model_label = str(body.get("model") or runtime.model_name)

        async with runtime._lock:
            out = await asyncio.to_thread(
                runtime.generate_single,
                prompt_text,
                max_tokens,
                temperature,
                steering,
            )

        comp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        if bool(body.get("stream", False)):
            words = out["text"].split(" ") if out["text"] else [""]

            async def _sse() -> Any:
                for idx, tok_piece in enumerate(words):
                    chunk_text = tok_piece if idx == 0 else (" " + tok_piece)
                    chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_label,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final_chunk = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_label,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "starbody": out["steering"],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        return JSONResponse({
            "id": comp_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_label,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out["text"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "total_tokens": out["prompt_tokens"] + out["completion_tokens"],
            },
            "starbody": {
                **out["steering"],
                "ms_per_tok": out["ms_per_tok"],
                "vram_peak_mb": out["vram_peak_mb"],
            },
        })

    @srv_app.post("/v1/completions")
    async def text_completions(request: Request, body: Dict[str, Any] = Body(...)) -> JSONResponse:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(status_code=400, detail="Field `prompt` must be a non-empty string.")
        steering = _resolve_steering(request, body)
        max_tokens = int(body.get("max_tokens", 48))
        temperature = float(body.get("temperature", 0.0))
        model_label = str(body.get("model") or runtime.model_name)

        async with runtime._lock:
            out = await asyncio.to_thread(
                runtime.generate_single,
                prompt,
                max_tokens,
                temperature,
                steering,
            )

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_label,
            "choices": [{"index": 0, "text": out["text"], "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "total_tokens": out["prompt_tokens"] + out["completion_tokens"],
            },
            "starbody": {
                **out["steering"],
                "ms_per_tok": out["ms_per_tok"],
                "vram_peak_mb": out["vram_peak_mb"],
            },
        })

    @srv_app.post("/v1/starbody/batch_completions")
    async def batch_completions(body: Dict[str, Any] = Body(...)) -> JSONResponse:
        prompts = body.get("prompts")
        doses = body.get("doses")
        cartridge = str(body.get("cartridge") or "").strip()
        if not isinstance(prompts, list) or not isinstance(doses, list) or not cartridge:
            raise HTTPException(status_code=400, detail="Provide `prompts` list, `doses` list, and `cartridge` name.")
        max_tokens = int(body.get("max_tokens", 24))
        pos_mode = str(body.get("position_mode", "all"))
        strict_bounds = bool(body.get("strict_bounds", False))

        try:
            async with runtime._lock:
                res = await asyncio.to_thread(
                    runtime.batch_generate_multitenant,
                    prompts,
                    doses,
                    cartridge,
                    max_tokens,
                    pos_mode,
                    strict_bounds,
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return JSONResponse(res)

    return srv_app

"""
Universal ModelAdapter Registry for Starbody / SMP v3.2.

Provides hardware-guarded activation capture, tuple-preserving forward hook output
normalization, strict float64 QR/SVD tangent-basis extraction, and multi-agent
syncytium trajectory adaptation for HuggingFace-style transformers and agent teams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from smp_core import (
    InteractomeLaplacianTracker,
    enforce_precision_and_seed,
)


HookTransformFn = Callable[[torch.Tensor], torch.Tensor]


class BaseModelAdapter(ABC):
    """Abstract base class bridging arbitrary neural/agent systems to SMP probes."""

    def __init__(self, model: Optional[nn.Module] = None) -> None:
        self.model = model

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Primary execution device of the adapted model."""

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Hidden dimension D along the final tensor axis."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of addressable decoder/transformer blocks."""

    @abstractmethod
    def get_layers(self) -> Sequence[nn.Module]:
        """Ordered sequence of addressable decoder blocks."""

    @abstractmethod
    def hook_path_prefix(self) -> str:
        """Module attribute path prefix (e.g., 'model.layers', 'gpt_neox.layers')."""

    def default_layer_indices(self, layer_fraction: float = 0.6) -> List[int]:
        """Selects upper-middle layer band covering roughly `layer_fraction` of depth."""
        n = self.num_layers
        if n <= 1:
            return [0]
        frac = max(0.1, min(1.0, float(layer_fraction)))
        count = max(1, int(round(n * frac)))
        start = max(0, (n - count) // 2 + 1 if n >= 4 else 0)
        end = min(n, start + count)
        return list(range(start, end))

    @staticmethod
    def normalize_hook_output(raw_output: Any, transform_fn: HookTransformFn) -> Any:
        """
        Applies `transform_fn` to the primary hidden-state Tensor [batch, seq_len, d_model]
        while preserving tuple/list/tensor output structure verbatim.
        """
        if isinstance(raw_output, tuple):
            if not raw_output:
                return raw_output
            primary = raw_output[0]
            if isinstance(primary, torch.Tensor):
                transformed = transform_fn(primary)
                return (transformed, *raw_output[1:])
            return raw_output
        if isinstance(raw_output, list):
            if not raw_output:
                return raw_output
            primary = raw_output[0]
            if isinstance(primary, torch.Tensor):
                transformed = transform_fn(primary)
                return [transformed, *raw_output[1:]]
            return raw_output
        if isinstance(raw_output, torch.Tensor):
            return transform_fn(raw_output)
        raise TypeError(f"Unsupported layer hook output type: {type(raw_output).__name__}")

    @abstractmethod
    def capture_activations(
        self,
        prompts: Union[Sequence[Any], torch.Tensor],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Captures float32 activations shaped [batch, seq_len, d_model] per layer."""

    @abstractmethod
    def extract_tangent_basis(
        self,
        positive_prompts: Sequence[Any],
        negative_prompts: Sequence[Any],
        layer_indices: Optional[Sequence[int]] = None,
        rank_k: int = 4,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Extracts orthonormal U_parallel [d_model, rank_k] & A_skew [rank_k, rank_k]."""


class HuggingFaceAdapter(BaseModelAdapter):
    """
    Adapter for HuggingFace decoder architectures and synthetic transformer blocks.
    Auto-introspects:
      - `model.model.layers`    (Qwen2 / Llama / Mistral / Gemma / SmolLM2)
      - `model.gpt_neox.layers` (Pythia / GPT-NeoX)
      - `model.transformer.h`   (GPT-2)
      - `model.layers`          (Generic synthetic HF-compatible transformer blocks)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model=model)
        self.tokenizer = tokenizer
        self._layers, self._hook_prefix = self._introspect_decoder_layers(model)
        self._device = device or self._infer_device(model)
        self._d_model = self._infer_hidden_size(model, self._layers)

    @staticmethod
    def _introspect_decoder_layers(model: nn.Module) -> Tuple[Sequence[nn.Module], str]:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            layers = inner.layers
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return list(layers), "model.layers"

        neox = getattr(model, "gpt_neox", None)
        if neox is not None and hasattr(neox, "layers"):
            layers = neox.layers
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return list(layers), "gpt_neox.layers"

        tr = getattr(model, "transformer", None)
        if tr is not None and hasattr(tr, "h"):
            layers = tr.h
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return list(layers), "transformer.h"

        if hasattr(model, "layers"):
            layers = model.layers
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return list(layers), "layers"

        raise ValueError(
            "Unsupported model architecture: expected decoder layers at "
            "`model.model.layers` (Qwen2/Llama/Mistral/Gemma/SmolLM2), "
            "`model.gpt_neox.layers` (Pythia), `model.transformer.h` (GPT-2), "
            "or `model.layers`."
        )

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        for param in model.parameters():
            return param.device
        for buf in model.buffers():
            return buf.device
        return torch.device("cpu")

    @staticmethod
    def _infer_hidden_size(model: nn.Module, layers: Sequence[nn.Module]) -> int:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "n_embd", "d_model"):
                val = getattr(cfg, attr, None)
                if isinstance(val, int) and val > 0:
                    return val
        for layer in layers:
            for p in layer.parameters():
                if p.ndim >= 1 and p.shape[-1] > 0:
                    return int(p.shape[-1])
        raise ValueError("Unable to infer d_model from model.config or decoder parameters.")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def get_layers(self) -> Sequence[nn.Module]:
        return self._layers

    def hook_path_prefix(self) -> str:
        return self._hook_prefix

    def resolve_layer_indices(self, layer_indices: Optional[Sequence[int]]) -> List[int]:
        if layer_indices is None:
            return list(range(self.num_layers))
        out: List[int] = []
        n = self.num_layers
        for idx in layer_indices:
            norm = idx if idx >= 0 else n + idx
            if norm < 0 or norm >= n:
                raise IndexError(f"Layer index {idx} out of range for num_layers={n}")
            if norm not in out:
                out.append(norm)
        return out

    def _encode_string_prompt(self, text: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.tokenizer is not None:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            return {k: v.to(self._device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}

        embed_fn = getattr(self.model, "get_input_embeddings", None)
        vocab_size = None
        if callable(embed_fn):
            emb = embed_fn()
            if emb is not None and hasattr(emb, "num_embeddings"):
                vocab_size = int(emb.num_embeddings)
        if vocab_size is None:
            cfg = getattr(self.model, "config", None)
            if cfg is not None and isinstance(getattr(cfg, "vocab_size", None), int):
                vocab_size = int(cfg.vocab_size)

        raw_bytes = list(text.encode("utf-8"))[:64] or [32]
        if vocab_size and vocab_size > 0 and self._expects_token_ids():
            ids = [(b + 1) % vocab_size for b in raw_bytes]
            return torch.tensor([ids], dtype=torch.long, device=self._device)  # [1, seq_len]

        # Continuous deterministic [1, seq_len, d_model] projection for embeddingless toy modules
        seq_len = min(len(raw_bytes), 16)
        vecs: List[torch.Tensor] = []
        for pos_idx in range(seq_len):
            b_val = raw_bytes[pos_idx]
            seed_bytes = hashlib.sha256(f"{pos_idx}:{b_val}".encode("utf-8")).digest()
            dims = torch.arange(self._d_model, dtype=torch.float32, device=self._device)
            phase = float(int.from_bytes(seed_bytes[:4], "little") % 1000) / 100.0
            v = torch.sin(0.37 * dims + phase) / math.sqrt(float(self._d_model))
            vecs.append(v)
        return torch.stack(vecs, dim=0).unsqueeze(0)  # [1, seq_len, d_model]

    def _expects_token_ids(self) -> bool:
        if hasattr(self.model, "get_input_embeddings") and callable(self.model.get_input_embeddings):
            if self.model.get_input_embeddings() is not None:
                return True
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Embedding) and ("embed" in name or "wte" in name or "tok" in name):
                return True
        return False

    def forward_item(self, prompt: Any) -> Any:
        """Executes a single forward pass on a Tensor, Dict, or string prompt without grads."""
        assert self.model is not None
        if isinstance(prompt, str):
            inp = self._encode_string_prompt(prompt)
        elif isinstance(prompt, torch.Tensor):
            t = prompt.to(self._device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            inp = t
        elif isinstance(prompt, dict):
            inp = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in prompt.items()}
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")

        if isinstance(inp, dict):
            return self.model(**inp)
        return self.model(inp)

    @torch.no_grad()
    def capture_activations(
        self,
        prompts: Union[Sequence[Any], torch.Tensor],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Captures float32 hidden states shaped [batch, seq_len, d_model] per target layer.
        All forward hooks are removed inside `try...finally`.
        """
        assert self.model is not None
        self.model.eval()
        self.model.requires_grad_(False)
        resolved = self.resolve_layer_indices(layer_indices)

        if isinstance(prompts, torch.Tensor):
            prompt_list: List[Any] = [prompts[i:i + 1] for i in range(prompts.shape[0])] if prompts.ndim >= 2 else [prompts]
        else:
            prompt_list = list(prompts)
        if not prompt_list:
            raise ValueError("prompts cannot be empty in capture_activations")

        collected: Dict[int, List[torch.Tensor]] = {idx: [] for idx in resolved}
        handles: List[Any] = []

        def _make_hook(l_idx: int) -> Callable[[nn.Module, Any, Any], None]:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                primary = out[0] if isinstance(out, (tuple, list)) else out
                if not isinstance(primary, torch.Tensor):
                    return
                act = primary.detach().to(dtype=torch.float32)
                if act.ndim == 2:
                    act = act.unsqueeze(0)  # [1, seq_len, d_model]
                collected[l_idx].append(act)

            return _hook

        try:
            for l_idx in resolved:
                handles.append(self._layers[l_idx].register_forward_hook(_make_hook(l_idx)))
            for item in prompt_list:
                _ = self.forward_item(item)
        finally:
            for h in handles:
                h.remove()

        result: Dict[int, torch.Tensor] = {}
        for l_idx, chunks in collected.items():
            if not chunks:
                raise RuntimeError(f"No activations captured at layer index {l_idx}.")
            max_seq = max(int(c.shape[1]) for c in chunks)
            aligned: List[torch.Tensor] = []
            for c in chunks:
                if c.shape[1] < max_seq:
                    pad_len = max_seq - int(c.shape[1])
                    pad = torch.zeros((c.shape[0], pad_len, c.shape[2]), dtype=torch.float32, device=c.device)
                    aligned.append(torch.cat([c, pad], dim=1))
                else:
                    aligned.append(c)
            result[l_idx] = torch.cat(aligned, dim=0).to(torch.float32)  # [batch, seq_len, d_model]
        return result

    @torch.no_grad()
    def extract_tangent_basis(
        self,
        positive_prompts: Sequence[Any],
        negative_prompts: Sequence[Any],
        layer_indices: Optional[Sequence[int]] = None,
        rank_k: int = 4,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extracts float32 orthonormal tangent bases U_parallel [d_model, rank_k] and
        skew-symmetric generators A_skew [rank_k, rank_k] using float64 SVD/QR.
        Verifies ||U^T U - I_k||_F < 1e-5 and ||A + A^T||_F < 1e-6 in float64.
        """
        enforce_precision_and_seed()
        if rank_k < 2:
            raise ValueError(f"rank_k must be >= 2 for nontrivial SO(k) rotations, got {rank_k}")

        resolved = self.resolve_layer_indices(layer_indices)
        pos_acts = self.capture_activations(positive_prompts, resolved)  # {l: [B_pos, S, D]}
        neg_acts = self.capture_activations(negative_prompts, resolved)  # {l: [B_neg, S, D]}

        u_by_layer: Dict[int, torch.Tensor] = {}
        a_by_layer: Dict[int, torch.Tensor] = {}

        for l_idx in resolved:
            p_l = pos_acts[l_idx]  # [B_p, S_p, D]
            n_l = neg_acts[l_idx]  # [B_n, S_n, D]
            n_pairs = min(int(p_l.shape[0]), int(n_l.shape[0]))
            if n_pairs <= 0:
                raise ValueError("Contrastive prompt sets must contain at least 1 pair.")

            # Flatten across matched token positions if pair count < rank_k so rank_k subspace is full-rank
            diff_last = (p_l[:n_pairs, -1, :] - n_l[:n_pairs, -1, :]).to(torch.float64)  # [N, D]
            min_seq = min(int(p_l.shape[1]), int(n_l.shape[1]))
            diff_all = (p_l[:n_pairs, :min_seq, :] - n_l[:n_pairs, :min_seq, :]).reshape(-1, self._d_model).to(torch.float64)
            diff_64 = torch.cat([diff_last, diff_all], dim=0)  # [N + N*S, D]

            if torch.norm(diff_64) < 1e-10:
                gen_seed = torch.Generator(device="cpu").manual_seed(20261117 + l_idx)
                diff_64 = torch.randn((max(rank_k * 2, 8), self._d_model), generator=gen_seed, dtype=torch.float64, device="cpu").to(self._device)

            _, svals, vh = torch.linalg.svd(diff_64.cpu(), full_matrices=False)  # vh: [r, D]
            cols: List[torch.Tensor] = []
            take_r = min(rank_k, int(vh.shape[0]))
            for j in range(take_r):
                cols.append(vh[j])

            if len(cols) < rank_k:
                gen = torch.Generator(device="cpu").manual_seed(9001 + l_idx)
                while len(cols) < rank_k:
                    cols.append(torch.randn(self._d_model, generator=gen, dtype=torch.float64))

            raw_mat_64 = torch.stack(cols, dim=1)  # [D, rank_k]
            q_64, r_mat_64 = torch.linalg.qr(raw_mat_64, mode="reduced")  # [D, rank_k]
            signs = torch.sign(torch.diag(r_mat_64))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            u_64 = (q_64 * signs.unsqueeze(0)).to(self._device)

            ident = torch.eye(rank_k, dtype=torch.float64, device=self._device)
            ortho_err = float(torch.linalg.norm(u_64.T @ u_64 - ident).item())
            if ortho_err >= 1e-5:
                raise AssertionError(f"Layer {l_idx} tangent basis non-orthonormal: err={ortho_err}")

            # Build skew-symmetric generator A [rank_k, rank_k] weighted by SVD ratios
            s_top = float(svals[0].item()) if len(svals) > 0 else 1.0
            s_top = max(s_top, 1e-12)
            a_64 = torch.zeros((rank_k, rank_k), dtype=torch.float64, device=self._device)
            a_64[0, 1] = 1.0
            a_64[1, 0] = -1.0
            if rank_k >= 4:
                sec_weight = max(0.25, min(0.85, float(svals[2].item() / s_top) if len(svals) > 2 else 0.5))
                a_64[2, 3] = sec_weight
                a_64[3, 2] = -sec_weight

            skew_err = float(torch.linalg.norm(a_64 + a_64.T).item())
            if skew_err >= 1e-6:
                raise AssertionError(f"Layer {l_idx} generator not skew-symmetric: err={skew_err}")

            u_by_layer[l_idx] = u_64.to(torch.float32)  # [d_model, rank_k]
            a_by_layer[l_idx] = a_64.to(torch.float32)  # [rank_k, rank_k]

        return u_by_layer, a_by_layer


class AgentProxyAdapter:
    """
    Lightweight async/sync wrapper projecting multi-agent turn logs (List[Dict[str, str]])
    into [M, D] embeddings and feeding `InteractomeLaplacianTracker` in `smp_core.py`
    for live Fiedler connectivity (lambda_2) and consensus exit-cost v_50(tau) tracking.
    """

    def __init__(
        self,
        agent_names: Optional[Sequence[str]] = None,
        state_dim: int = 64,
        embed_fn: Optional[Callable[[str], torch.Tensor]] = None,
        g_max: float = 1.4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent_names = list(agent_names or ["Planner", "Coder", "Auditor", "Synthesizer"])
        self.num_nodes = len(self.agent_names)
        self.state_dim = int(state_dim)
        self.embed_fn = embed_fn
        self.device = device or torch.device("cpu")

        # Default tangent basis spanning consensus-to-deviation mode
        idx = torch.arange(self.state_dim, dtype=torch.float64, device=self.device)
        v0 = torch.cos(0.25 * idx)
        v0 = v0 / torch.norm(v0).clamp(min=1e-12)
        v1 = torch.sin(0.50 * idx)
        v1 = v1 - (v1 @ v0) * v0
        v1 = v1 / torch.norm(v1).clamp(min=1e-12)
        self.tangent_basis = torch.stack([v0, v1], dim=1).to(torch.float32)  # [D, 2]

        scale_d = math.sqrt(float(self.state_dim))
        self.tracker = InteractomeLaplacianTracker(
            num_nodes=self.num_nodes,
            state_dim=self.state_dim,
            tangent_basis=self.tangent_basis,
            g_max=g_max,
            v_half=0.85 * scale_d,
            v_slope=0.18 * scale_d,
            dt=0.15,
            anchor_gamma=1.0,
            beta_consensus=2.2,
        ).to(self.device)
        self._baseline_nodes: Optional[torch.Tensor] = None
        self._baseline_lambda2: Optional[float] = None
        self._turn_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clears baseline reference & recorded turn history for a new session."""
        self._baseline_nodes = None
        self._baseline_lambda2 = None
        self._turn_history = []

    def _measure_emergent_v50(self, x_nodes: torch.Tensor, delay_turns: int) -> float:
        """
        Pure-emergent bisection using `InteractomeLaplacianTracker.apply_counter_intervention_after_delay`
        to compute the exact counter-dose / write-dose ratio needed to reverse node-0 drift.
        """
        assert self._baseline_nodes is not None
        mean_shift = (x_nodes - self._baseline_nodes).to(torch.float64)
        shift_norm = float(torch.linalg.norm(mean_shift).item())
        if shift_norm < 1e-6:
            return 1.0

        diff0 = mean_shift[0]
        d0_norm = float(torch.linalg.norm(diff0).item())
        drive_raw = diff0 if d0_norm >= 1e-6 else mean_shift.mean(dim=0)
        drive_norm = float(torch.linalg.norm(drive_raw).clamp(min=1e-12).item())
        drive_unit = (drive_raw / drive_norm).to(dtype=x_nodes.dtype, device=self.device)

        write_dose = max(0.3, min(1.5, d0_norm if d0_norm >= 1e-6 else shift_norm))
        tau = max(1, min(16, int(delay_turns)))

        x_pos = self.tracker.apply_counter_intervention_after_delay(
            self._baseline_nodes,
            drive_unit,
            write_dose=write_dose,
            delay_turns=tau,
            counter_dose=0.0,
            write_turns=4,
            counter_turns=4,
            settle_turns=4,
        )
        target_proj = float(
            torch.dot((x_pos[0] - self._baseline_nodes[0]).to(torch.float64), drive_unit.to(torch.float64)).item()
        )
        if target_proj <= 1e-6:
            return 1.0
        half_proj = 0.5 * target_proj

        lo, hi = 0.0, 4.0 * write_dose
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            x_try = self.tracker.apply_counter_intervention_after_delay(
                self._baseline_nodes,
                drive_unit,
                write_dose=write_dose,
                delay_turns=tau,
                counter_dose=mid,
                write_turns=4,
                counter_turns=4,
                settle_turns=4,
            )
            cur_proj = float(
                torch.dot((x_try[0] - self._baseline_nodes[0]).to(torch.float64), drive_unit.to(torch.float64)).item()
            )
            if cur_proj <= half_proj:
                hi = mid
            else:
                lo = mid
        return round(float((0.5 * (lo + hi)) / max(write_dose, 1e-12)), 4)

    def _encode_message(self, text: str) -> torch.Tensor:
        if self.embed_fn is not None:
            v = self.embed_fn(text).to(dtype=torch.float32, device=self.device).reshape(-1)
            if v.shape[0] != self.state_dim:
                raise ValueError(f"Expected embed_fn dim {self.state_dim}, got {v.shape[0]}")
            return v
        # Deterministic hashed spectral projection -> [state_dim]
        dims = torch.arange(self.state_dim, dtype=torch.float64, device=self.device)
        accum = torch.zeros(self.state_dim, dtype=torch.float64, device=self.device)
        tokens = [t for t in str(text).lower().split() if t] or ["empty"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            freq = 0.15 + float(int.from_bytes(digest[:2], "little") % 64) * 0.03
            phase = float(int.from_bytes(digest[2:4], "little") % 360) * math.pi / 180.0
            accum += torch.cos(freq * dims + phase)
        norm = torch.norm(accum).clamp(min=1e-12)
        return (accum / norm * math.sqrt(float(self.state_dim))).to(torch.float32)  # [D]

    def project_turn_logs(self, turn_logs: Sequence[Dict[str, str]]) -> torch.Tensor:
        """
        Projects agent messages from a round into [M, D] float32 embeddings.
        Each turn dict contains keys like `{"agent": "Planner", "content": "..."}`.
        """
        by_agent: Dict[str, str] = {name: "" for name in self.agent_names}
        for item in turn_logs:
            ag = str(item.get("agent", item.get("role", ""))).strip()
            msg = str(item.get("content", item.get("text", "")))
            if ag in by_agent:
                by_agent[ag] = (by_agent[ag] + " " + msg).strip()
        rows = [self._encode_message(by_agent[name]) for name in self.agent_names]
        return torch.stack(rows, dim=0)  # [M, D]

    @torch.no_grad()
    def evaluate_turns(
        self,
        turn_logs: Sequence[Dict[str, str]],
        turns_horizon: int = 8,
    ) -> Dict[str, Any]:
        """Feeds [M, D] state matrix through InteractomeLaplacianTracker and returns metrics."""
        x_nodes = self.project_turn_logs(turn_logs)  # [M, D]
        if self._baseline_nodes is None:
            self._baseline_nodes = x_nodes.clone()
        g_mat, _, stats = self.tracker.compute_conductance_and_laplacian(x_nodes)
        lam2 = float(stats["fiedler_lambda2"])
        if self._baseline_lambda2 is None:
            self._baseline_lambda2 = max(lam2, 1e-6)

        mean_shift = x_nodes - self._baseline_nodes
        e_total, e_parallel, e_perp = self.tracker.certifier.decompose_state_shift(mean_shift)

        lam2_ratio = float(lam2 / max(self._baseline_lambda2, 1e-6))
        if lam2_ratio < 0.45 or lam2 < 0.25:
            fracture_risk = "FRACTURE_IMMINENT"
        elif lam2_ratio < 0.80:
            fracture_risk = "TREMOR_WARNING"
        else:
            fracture_risk = "NOMINAL"

        tau_ref = max(1, int(turns_horizon))
        v50_emergent = self._measure_emergent_v50(x_nodes, delay_turns=tau_ref)
        turn_idx = len(self._turn_history)
        out = {
            "turn_index": turn_idx,
            "agent_names": list(self.agent_names),
            "node_embeddings_shape": list(x_nodes.shape),
            "fiedler_lambda2": lam2,
            "lambda2_ratio": round(lam2_ratio, 4),
            "fracture_risk": fracture_risk,
            "mean_conductance": float(stats["mean_conductance"]),
            "conductance_matrix": g_mat.detach().cpu().tolist(),
            "e_total": float(e_total),
            "e_parallel": float(e_parallel),
            "e_perp": float(e_perp),
            "consensus_v50_tau": float(v50_emergent),
            "tau_horizon": tau_ref,
        }
        self._turn_history.append(out)
        return out

    async def evaluate_turns_async(
        self,
        turn_logs: Sequence[Dict[str, str]],
        turns_horizon: int = 8,
    ) -> Dict[str, Any]:
        """Async middleware entrypoint for streaming multi-agent turn telemetry."""
        return self.evaluate_turns(turn_logs=turn_logs, turns_horizon=turns_horizon)

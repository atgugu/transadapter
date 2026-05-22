from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transadapter.blocks.feedback import FeedbackProjection
from transadapter.blocks.mor_router import MoRRouter
from transadapter.blocks.specialists import ParallelSpecialists
from transadapter.blocks.workspace import Workspace
from transadapter.config import TransAdapterConfig


@dataclass
class TransAdapterOutput:
    """Container for forward outputs.

    `aux` collects per-block diagnostics for mech-interp; e.g. halt probabilities,
    keep weights, write assignments. Tagged by block index.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None
    aux: dict


class _MLP(nn.Module):
    def __init__(self, config: TransAdapterConfig, mult: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, mult * config.d_model, bias=config.bias)
        self.proj = nn.Linear(mult * config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc(x)))


class TransAdapterBlock(nn.Module):
    """One TransAdapter block: parallel specialists -> workspace write -> broadcast read
    -> feedback -> MLP, recursively iterated up to max_steps with per-token soft halt.

    The workspace W is threaded through the whole forward pass — this block reads
    and updates it but does not own it.
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.specialists = ParallelSpecialists(config)
        self.router = MoRRouter(config)
        self.feedback = FeedbackProjection(config)
        self.norm_1 = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
        self.norm_2 = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
        self.mlp = _MLP(config)

    def forward(
        self, stream: torch.Tensor, W: torch.Tensor, workspace: Workspace, train_progress: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the block with per-token soft adaptive recursion.

        Returns (new_stream, new_W, aux) where aux contains per-step halt probs,
        keep weights, and write assignments for diagnostics.
        """
        S = self.config.max_steps
        halt_probs: list[torch.Tensor] = []
        streams_per_step: list[torch.Tensor] = []
        assigns_per_step: list[torch.Tensor] = []

        s_curr = stream
        W_curr = W
        for step in range(S):
            halt_logit = self.router.step_logits(s_curr, W_curr)
            halt_p = torch.sigmoid(halt_logit).clamp(self.config.halt_eps, 1 - self.config.halt_eps)
            halt_probs.append(halt_p)

            x = self.norm_1(s_curr)
            stream_update, candidates, salience = self.specialists(x)
            W_curr, assign = workspace.write(W_curr, candidates, salience)
            broadcast = workspace.broadcast_read(x, W_curr)
            feedback_delta = self.feedback(x, W_curr, train_progress)

            s_after_attn = s_curr + stream_update + broadcast + feedback_delta
            s_after_mlp = s_after_attn + self.mlp(self.norm_2(s_after_attn))
            s_curr = s_after_mlp

            streams_per_step.append(s_curr)
            assigns_per_step.append(assign)

        halt_probs_T = torch.stack(halt_probs, dim=0)
        keep, remainder = self.router.expected_depth_weights(halt_probs_T, eps=self.config.halt_eps)
        streams_T = torch.stack(streams_per_step, dim=0)
        new_stream = (keep.unsqueeze(-1) * streams_T).sum(dim=0)

        aux = {
            "halt_probs": halt_probs_T.detach(),
            "keep_weights": keep.detach(),
            "remainder": remainder.detach(),
            "write_assigns": torch.stack(assigns_per_step, dim=0).detach(),
        }
        return new_stream, W_curr, aux


class TransAdapter(nn.Module):
    """TransAdapter: small persistent latent + adaptive recursion + feedback."""

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.workspace = Workspace(config)
        self.blocks = nn.ModuleList(
            [TransAdapterBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
        if config.tie_embeddings:
            self.lm_head = lambda x: x @ self.tok_emb.weight.T  # type: ignore[assignment]
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        train_progress: float = 1.0,
        return_aux: bool = False,
    ) -> TransAdapterOutput:
        """idx: [B, L] int token ids. Returns logits [B, L, V] and optional CE loss."""
        B, L = idx.shape
        assert L <= self.config.max_seq_len, f"sequence length {L} > max {self.config.max_seq_len}"
        pos = torch.arange(L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        W = self.workspace.init_state(B, idx.device, x.dtype)
        per_block_aux = []
        for i, block in enumerate(self.blocks):
            x, W, aux = block(x, W, self.workspace, train_progress=train_progress)
            if return_aux:
                per_block_aux.append({"block": i, **aux})

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return TransAdapterOutput(
            logits=logits,
            loss=loss,
            aux={"blocks": per_block_aux, "workspace_final": W.detach() if return_aux else None},
        )

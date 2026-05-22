from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transadapter.config import TransAdapterConfig


class Workspace(nn.Module):
    """Persistent latent W in R^{N x d_w}, written by competing specialists.

    Q1 architectural commitment: W is a single object across the entire forward
    pass. The MoRRouter reads from W via cross-attention (`broadcast_read`), so
    halting dynamics and write-gate dynamics share an input subspace.

    Functional roles:
      - meta_init: learned initial state per slot (Hymba-style).
      - write(candidates, salience): top-k competition over (specialist, slot)
        pairs followed by a gated update.
      - broadcast_read(stream): cross-attend stream queries to workspace
        keys/values.
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        N, d_w, K = config.n_slots, config.d_workspace, config.n_specialists

        self.meta_init = nn.Parameter(torch.randn(N, d_w) * 0.02)

        self.alpha = nn.Parameter(
            torch.full((N,), torch.logit(torch.tensor(config.workspace_alpha_init)).item())
        )

        self.slot_keys = nn.Parameter(torch.randn(N, d_w) * (1.0 / d_w**0.5))

        self.read_q = nn.Linear(config.d_model, d_w, bias=config.bias)
        self.read_k = nn.Linear(d_w, d_w, bias=config.bias)
        self.read_v = nn.Linear(d_w, d_w, bias=config.bias)
        self.read_o = nn.Linear(d_w, config.d_model, bias=config.bias)

        self.norm_w = nn.LayerNorm(d_w, eps=config.norm_eps, bias=config.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.meta_init.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(dtype)

    def write(
        self,
        W: torch.Tensor,
        candidates: torch.Tensor,
        salience: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update W with top-k competitive writes.

        Args:
          W: [B, N, d_w] current workspace.
          candidates: [B, K, d_w] write proposals from K specialists (pooled across positions).
          salience: [B, K] per-specialist write priority.

        Returns:
          W_next: [B, N, d_w] updated workspace.
          assign: [B, K, N] soft assignment used for top-k write (for probes/diagnostics).
        """
        B, N, d_w = W.shape
        K = candidates.shape[1]

        affinity = torch.einsum("bkd,nd->bkn", candidates, self.slot_keys) / (d_w**0.5)
        affinity = affinity + salience.unsqueeze(-1)

        top_k = self.config.top_k_write
        flat = rearrange(affinity, "b k n -> b (k n)")
        topk_vals, topk_idx = flat.topk(top_k, dim=-1)
        mask = torch.full_like(flat, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        gated = F.softmax(mask, dim=-1)
        assign = rearrange(gated, "b (k n) -> b k n", k=K, n=N)

        delta = torch.einsum("bkn,bkd->bnd", assign, candidates)

        alpha = torch.sigmoid(self.alpha).view(1, N, 1)
        W_next = (1.0 - alpha) * W + alpha * delta
        W_next = self.norm_w(W_next)
        return W_next, assign

    def broadcast_read(self, stream: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Cross-attend stream queries to workspace.

        Args:
          stream: [B, L, d_model]
          W: [B, N, d_w]

        Returns:
          [B, L, d_model] update to be added back into the stream.
        """
        d_w = self.config.d_workspace
        q = self.read_q(stream)
        k = self.read_k(W)
        v = self.read_v(W)

        attn = torch.einsum("bld,bnd->bln", q, k) / (d_w**0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bln,bnd->bld", attn, v)
        return self.read_o(out)

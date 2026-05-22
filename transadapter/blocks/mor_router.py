from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transadapter.config import TransAdapterConfig


class MoRRouter(nn.Module):
    """Per-token recursion halt router.

    The Q1 architectural bet: input to the halt head is `[S_t ; attn(S_t, W)]`,
    forcing the halting subspace to share inputs with the workspace-write
    subspace. Set `config.enable_router_reads_workspace = False` for the ablation
    that removes the workspace dependence (router reads only the stream).

    Phase 0 uses an *expected-depth* (soft) formulation: every token runs
    `max_steps` block iterations, and the per-step contribution is weighted by
    `(1 - cumulative halt prob)`. This is differentiable and batches cleanly,
    while preserving the per-token halt distribution as a measurable object.

    Outputs at each step:
      - p_halt_step: per-token halt probability at this step ([B, L]).
      - keep_weight: per-token weight to apply to this step's contribution
        ([B, L]); summing keep_weights across steps equals 1.
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.d_model + (config.d_workspace if config.enable_router_reads_workspace else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, config.router_hidden, bias=config.bias),
            nn.SiLU(),
            nn.Linear(config.router_hidden, 1, bias=config.bias),
        )
        self.read_q = nn.Linear(config.d_model, config.d_workspace, bias=config.bias)

    def workspace_attn(self, stream: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Per-token cross-attention from stream to W; output is [B, L, d_w]."""
        d_w = self.config.d_workspace
        q = self.read_q(stream)
        attn = torch.einsum("bld,bnd->bln", q, W) / (d_w**0.5)
        attn = F.softmax(attn, dim=-1)
        return torch.einsum("bln,bnd->bld", attn, W)

    def step_logits(self, stream: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Halt logits per token at the current step, shape [B, L]."""
        if self.config.enable_router_reads_workspace:
            ws = self.workspace_attn(stream, W)
            inp = torch.cat([stream, ws], dim=-1)
        else:
            inp = stream
        return self.head(inp).squeeze(-1) / self.config.halt_temperature

    @staticmethod
    def expected_depth_weights(
        halt_probs: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert per-step halt probs to per-step keep-weights.

        Args:
          halt_probs: [steps, B, L] in (0, 1).
          eps: minimum residual probability after final step (will be added so the
               weights sum to 1 even with hard halt at the last step).

        Returns:
          keep_weights: [steps, B, L], sum across steps == 1.
          remainder: [B, L] cumulative non-halt probability after the last step.
        """
        S = halt_probs.shape[0]
        not_halt = 1.0 - halt_probs
        cum_not_halt = torch.cumprod(not_halt, dim=0)
        prev = torch.cat([torch.ones_like(cum_not_halt[:1]), cum_not_halt[:-1]], dim=0)
        keep = halt_probs * prev
        # mass remaining after last step is cum_not_halt[-1]; assign it to the final step
        remainder = cum_not_halt[-1]
        keep[-1] = keep[-1] + remainder
        # epsilon-floor regularization: blend with uniform 1/S to keep gradients alive
        if eps > 0:
            keep = (1.0 - eps) * keep + eps / S
            keep = keep / keep.sum(dim=0, keepdim=True)
        return keep, remainder

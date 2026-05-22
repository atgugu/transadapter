from __future__ import annotations

import torch
import torch.nn as nn

from transadapter.config import TransAdapterConfig


class FeedbackProjection(nn.Module):
    """Top-down low-rank projection from W back into the stream.

    Functionally: after a block computes its updated W, the next block (or the
    same block on a future forward pass) can consume `feedback(stream, W)` added
    into its input. The gate is warmed up from 0 to a learned value to avoid the
    early-feedback instability that kills naive Feedback-Transformer setups.

    For this implementation we project W -> stream-aligned correction at the
    *current* layer rather than the *previous* layer; the more aggressive
    previous-layer routing is a Phase 1 follow-up that requires unrolling the
    forward pass. The functional content (low-rank top-down correction with
    gate warmup) is preserved.
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        r = config.feedback_rank
        self.k_fb = nn.Parameter(torch.randn(config.d_workspace, r) * (1.0 / config.d_workspace**0.5))
        self.proj = nn.Linear(r, config.d_model, bias=config.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def gate(self, train_progress: float) -> torch.Tensor:
        """Compute the warmup-modulated feedback gate.

        train_progress in [0, 1]; ramps the gate from 0 to its learned value over
        `config.feedback_warmup_frac` of training, then linearly to 1x learned
        over the next 2x that fraction.
        """
        if not self.config.enable_feedback:
            return torch.zeros((), device=self.gate_logit.device, dtype=self.gate_logit.dtype)
        warm = self.config.feedback_warmup_frac
        if warm <= 0 or train_progress <= 0:
            return torch.sigmoid(self.gate_logit)
        if train_progress < warm:
            return torch.zeros_like(self.gate_logit)
        ramp_end = min(1.0, 3.0 * warm)
        if train_progress < ramp_end:
            scale = (train_progress - warm) / max(ramp_end - warm, 1e-6)
            return scale * torch.sigmoid(self.gate_logit)
        return torch.sigmoid(self.gate_logit)

    def forward(
        self, stream: torch.Tensor, W: torch.Tensor, train_progress: float = 1.0
    ) -> torch.Tensor:
        """Compute gated low-rank feedback contribution to add into the stream.

        Args:
          stream: [B, L, d_model]
          W:      [B, N, d_w]
          train_progress: [0,1] for warmup gate; defaults to 1 (post-warmup).

        Returns:
          [B, L, d_model]
        """
        gate = self.gate(train_progress)
        if float(gate.detach()) == 0.0:
            return torch.zeros_like(stream)
        # mean-pool over slots, project low-rank, broadcast across positions
        wbar = W.mean(dim=1)
        low = wbar @ self.k_fb
        delta = self.proj(low).unsqueeze(1).expand_as(stream)
        return gate * delta

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transadapter.config import TransAdapterConfig


def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


def _sliding_mask(L: int, window: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    return (diff > 0) | (-diff > window)


class CausalAttention(nn.Module):
    """Standard multi-head causal attention. Specialist 1: 'local snapshot memory.'"""

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.config.n_heads, self.config.head_dim
        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        mask = _causal_mask(L, x.device)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class SlidingWindowAttention(nn.Module):
    """Specialist 2: bounded-context retrieval head."""

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.config.n_heads, self.config.head_dim
        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        mask = _sliding_mask(L, self.config.sliding_window, x.device)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class LiteSSM(nn.Module):
    """Specialist 3: simple gated linear recurrence as a Mamba-stand-in for Phase 0.

    A real Mamba block can be substituted for this in Phase 1; the interface (B,L,D
    in / out) is identical. We keep this dependency-light so Phase 0 runs on plain
    PyTorch without custom CUDA kernels.

    State dimension is `config.ssm_state_dim` per channel. The recurrence is
    h_t = sigmoid(g_t) * h_{t-1} + (1 - sigmoid(g_t)) * v_t, output is a learned
    projection of [v_t ; h_t].
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        S = config.ssm_state_dim
        self.in_proj = nn.Linear(config.d_model, 2 * S, bias=config.bias)
        self.gate = nn.Linear(config.d_model, S, bias=config.bias)
        self.out_proj = nn.Linear(2 * S, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        S = self.config.ssm_state_dim
        v = self.in_proj(x).view(B, L, 2, S)
        v_in, v_skip = v[:, :, 0], v[:, :, 1]
        g = torch.sigmoid(self.gate(x))

        h = torch.zeros(B, S, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = g[:, t] * h + (1.0 - g[:, t]) * v_in[:, t]
            outs.append(h)
        H = torch.stack(outs, dim=1)
        cat = torch.cat([H, v_skip], dim=-1)
        return self.out_proj(cat)


class ParallelSpecialists(nn.Module):
    """K specialists running in parallel on the stream.

    Each specialist produces (a) a stream-update that is added to the residual
    (Hymba-style: averaged across specialists), and (b) a single 'write candidate'
    + salience for the workspace top-k competition.

    The stream-update keeps the architecture functional even if W is irrelevant
    (R3 mitigation: the residual bottleneck pushes information *through* W, but
    the specialists still contribute directly so the model can train).
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        assert config.n_specialists == 3, "Phase 0 wires exactly 3 specialists"

        self.local_attn = CausalAttention(config)
        self.window_attn = SlidingWindowAttention(config)
        self.ssm = LiteSSM(config)

        # write head per specialist: pool stream -> (candidate, salience)
        d_w = config.d_workspace
        self.to_write = nn.ModuleList(
            [nn.Linear(config.d_model, d_w + 1, bias=config.bias) for _ in range(3)]
        )

    def _pool(self, stream_out: torch.Tensor) -> torch.Tensor:
        # mean over positions; cheap and parameter-free pooling for write candidates
        return stream_out.mean(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (stream_update, candidates, salience).

        stream_update: [B, L, d_model] mean of specialist outputs.
        candidates:    [B, K, d_w]
        salience:      [B, K]
        """
        outs = [self.local_attn(x), self.window_attn(x), self.ssm(x)]
        stream_update = torch.stack(outs, dim=0).mean(dim=0)

        cands_sals = []
        for i, o in enumerate(outs):
            pooled = self._pool(o)
            cs = self.to_write[i](pooled)
            cands_sals.append(cs)
        cs = torch.stack(cands_sals, dim=1)
        candidates = cs[..., :-1]
        salience = cs[..., -1]
        return stream_update, candidates, salience

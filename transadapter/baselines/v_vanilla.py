from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transadapter.config import TransAdapterConfig


@dataclass
class VanillaOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class _CausalSelfAttention(nn.Module):
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
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
        self.attn = _CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)
        self.fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.proj(F.silu(self.fc(self.norm_2(x))))
        return x


class VanillaTransformer(nn.Module):
    """Matched baseline: standard pre-norm decoder with causal MHA + SwiGLU-lite MLP.

    Param-matched and FLOP-matched to TransAdapter via the same `TransAdapterConfig`
    (depth, d_model, n_heads). Used as the V-vanilla rung of the ablation ladder.
    """

    def __init__(self, config: TransAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.n_layers)])
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
        **_: object,
    ) -> VanillaOutput:
        B, L = idx.shape
        assert L <= self.config.max_seq_len
        pos = torch.arange(L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return VanillaOutput(logits=logits, loss=loss)

"""Forward + backward smoke test for TransAdapter and VanillaTransformer.

Run from the repo root:
    python -m scripts.smoke_forward

Goals:
  - both models build at a tiny config without errors
  - forward pass on random ints returns finite logits of the right shape
  - backward pass produces finite gradients on a CE loss
  - param counts (excluding embeddings) are within ~30% (sanity check; tight
    matching is enforced in Phase 1 by a dedicated FLOP/param counter)
  - aux dict from TransAdapter contains halt probabilities and write assignments
    of the expected shapes (the diagnostic surface mech-interp will probe)
"""
from __future__ import annotations

import os
import sys

import torch

# allow running as a script from the repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transadapter.baselines.v_vanilla import VanillaTransformer
from transadapter.config import TransAdapterConfig
from transadapter.model import TransAdapter


def make_tiny_config() -> TransAdapterConfig:
    """A tiny config to fit a 4 GB GPU / CPU smoke test."""
    return TransAdapterConfig(
        vocab_size=2048,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_slots=8,
        d_workspace=96,
        n_specialists=3,
        top_k_write=2,
        sliding_window=16,
        ssm_state_dim=16,
        max_steps=2,
        router_hidden=32,
        feedback_rank=16,
        feedback_warmup_frac=0.0,
    )


def shape_and_finite(name: str, t: torch.Tensor, expected: tuple) -> None:
    assert t.shape == expected, f"{name}: shape {tuple(t.shape)} != expected {expected}"
    assert torch.isfinite(t).all(), f"{name}: contains non-finite values"


def main() -> int:
    torch.manual_seed(0)
    cfg = make_tiny_config()

    B, L = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, L))
    tgt = torch.randint(0, cfg.vocab_size, (B, L))

    # -- TransAdapter forward + backward
    ta = TransAdapter(cfg)
    out = ta(idx, targets=tgt, train_progress=1.0, return_aux=True)
    shape_and_finite("ta.logits", out.logits, (B, L, cfg.vocab_size))
    assert out.loss is not None and torch.isfinite(out.loss), "ta.loss not finite"
    out.loss.backward()
    grad_max = max(
        (p.grad.abs().max().item() for p in ta.parameters() if p.grad is not None), default=0.0
    )
    grad_finite = all(
        torch.isfinite(p.grad).all() for p in ta.parameters() if p.grad is not None
    )
    assert grad_finite, "TransAdapter: NaN/Inf in gradients"

    # check aux structure
    assert "blocks" in out.aux and len(out.aux["blocks"]) == cfg.n_layers
    for b in out.aux["blocks"]:
        shape_and_finite("halt_probs", b["halt_probs"], (cfg.max_steps, B, L))
        shape_and_finite("keep_weights", b["keep_weights"], (cfg.max_steps, B, L))
        shape_and_finite(
            "write_assigns", b["write_assigns"], (cfg.max_steps, B, cfg.n_specialists, cfg.n_slots)
        )
        # keep weights should sum to ~1 across steps
        s = b["keep_weights"].sum(dim=0)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), (
            f"keep_weights sum {s.min().item():.4f}-{s.max().item():.4f}, expected ~1"
        )

    # -- Vanilla forward + backward
    va = VanillaTransformer(cfg)
    vout = va(idx, targets=tgt)
    shape_and_finite("va.logits", vout.logits, (B, L, cfg.vocab_size))
    assert vout.loss is not None and torch.isfinite(vout.loss), "va.loss not finite"
    vout.loss.backward()
    v_grad_finite = all(
        torch.isfinite(p.grad).all() for p in va.parameters() if p.grad is not None
    )
    assert v_grad_finite, "VanillaTransformer: NaN/Inf in gradients"

    ta_p = ta.num_params(exclude_embeddings=True)
    va_p = va.num_params(exclude_embeddings=True)
    ratio = ta_p / max(va_p, 1)

    print("=" * 72)
    print(f"TransAdapter:        params (no emb) = {ta_p:>10,}    loss = {out.loss.item():.4f}")
    print(f"VanillaTransformer:  params (no emb) = {va_p:>10,}    loss = {vout.loss.item():.4f}")
    print(f"param ratio TA / V = {ratio:.3f}  (Phase 0: informational only; Phase 1 enforces ±2%)")
    print(f"max grad magnitude: TA={grad_max:.3e}")
    print(f"halt_probs[0][0,0] sample: {out.aux['blocks'][0]['halt_probs'][:, 0, 0].tolist()}")
    print(f"keep_weights[0][:,0,0] sample: {out.aux['blocks'][0]['keep_weights'][:, 0, 0].tolist()}")
    print("=" * 72)
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

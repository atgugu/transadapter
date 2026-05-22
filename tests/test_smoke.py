"""Smoke + ablation tests for the architecture.

These do not test capability — they test that the architecture compiles, runs,
and exposes the diagnostic surface mech-interp will probe.
"""
from __future__ import annotations

import torch

from transadapter.baselines.v_vanilla import VanillaTransformer
from transadapter.config import TransAdapterConfig
from transadapter.model import TransAdapter


def tiny_config(**overrides) -> TransAdapterConfig:
    base = dict(
        vocab_size=512,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_slots=4,
        d_workspace=48,
        n_specialists=3,
        top_k_write=2,
        sliding_window=8,
        ssm_state_dim=8,
        max_steps=2,
        router_hidden=16,
        feedback_rank=8,
        feedback_warmup_frac=0.0,
    )
    base.update(overrides)
    return TransAdapterConfig(**base)


def _forward_backward(model, idx, tgt) -> torch.Tensor:
    out = model(idx, targets=tgt)
    out.loss.backward()
    return out.loss


def test_transadapter_forward_backward():
    torch.manual_seed(1)
    cfg = tiny_config()
    model = TransAdapter(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss = _forward_backward(model, idx, tgt)
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_vanilla_forward_backward():
    torch.manual_seed(1)
    cfg = tiny_config()
    model = VanillaTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss = _forward_backward(model, idx, tgt)
    assert torch.isfinite(loss)


def test_router_decoupled_ablation():
    """Q1 ablation: router reads only the stream, not W."""
    torch.manual_seed(2)
    cfg = tiny_config(enable_router_reads_workspace=False)
    model = TransAdapter(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss = _forward_backward(model, idx, tgt)
    assert torch.isfinite(loss)
    # router input dim should be d_model (not d_model + d_workspace)
    first = model.blocks[0].router.head[0]
    assert first.in_features == cfg.d_model


def test_feedback_disabled_ablation():
    """V-workspace-mor: same model, no feedback."""
    torch.manual_seed(3)
    cfg = tiny_config(enable_feedback=False)
    model = TransAdapter(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss = _forward_backward(model, idx, tgt)
    assert torch.isfinite(loss)


def test_keep_weights_sum_to_one():
    """The expected-depth formulation must produce keep weights that sum to 1
    across recursion steps for every (batch, position)."""
    torch.manual_seed(4)
    cfg = tiny_config(max_steps=4)
    model = TransAdapter(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(idx, return_aux=True)
    for blk in out.aux["blocks"]:
        s = blk["keep_weights"].sum(dim=0)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), s


def test_workspace_persistence():
    """W must be threaded across all blocks; the final-block W must reflect updates."""
    torch.manual_seed(5)
    cfg = tiny_config()
    model = TransAdapter(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    out = model(idx, return_aux=True)
    W_final = out.aux["workspace_final"]
    W_init = model.workspace.init_state(2, idx.device, W_final.dtype)
    diff = (W_final - W_init).abs().max().item()
    assert diff > 1e-3, f"workspace not updated; max diff = {diff}"

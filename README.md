# TransAdapter

A small-model architecture probe testing whether **(persistent latent workspace) + (per-token adaptive recursion depth) + (top-down feedback)** compute *better together than the sum of their parts* — and whether the per-token halt router and the workspace write-gate share a representational subspace.

All claims, metrics, and decision rules are functional, mechanistic, and measurable on the trained model. Biological framing is informative only; nothing in the evaluation pipeline depends on it.

## Status

This repository currently contains the **architecture and a forward/backward smoke test**. Training, evaluation, and mech-interp are not yet implemented.

| Component | Status |
|---|---|
| `transadapter/` model + blocks (workspace, MoR router, parallel specialists, feedback) | implemented |
| `transadapter/baselines/v_vanilla.py` (matched-shape vanilla transformer) | implemented |
| `scripts/smoke_forward.py` | implemented |
| `tests/` (forward/backward, ablation toggles, keep-weight invariant, workspace persistence) | implemented |
| Matched FLOPs/params counter | not yet |
| Training script + data pipeline | not yet |
| `eval/` (broadcast events, codim overlap, controller ID, circuits) | not yet |
| `saes/` (SAE features for Q1 subspace test) | not yet |
| `configs/`, `preregistration/`, `papers/` (positive + null outlines) | not yet |

## Quick start

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# forward/backward smoke test (CPU is fine)
python -m scripts.smoke_forward

# unit tests
pytest
```

## Architecture (one block)

```
parallel specialists  (causal attn | sliding-window attn | lite-SSM)
        |
        +-- stream update  ----------------------------> residual
        +-- write candidates + salience
                                  |
                          workspace.write(W, candidates, salience)   # top-k competitive write
                                  |
                          workspace.broadcast_read(stream, W)        # cross-attn back to stream
                                  |
                          feedback(stream, W, train_progress)        # low-rank, warmup-gated
                                  |
                                 MLP
                                  |
                          MoR router halts per-token  -->  expected-depth soft halting
                                                           (keep-weights sum to 1 across steps)
```

`W` is a single object threaded through the entire forward pass. The MoR router reads `[stream ; cross-attn(stream, W)]`, which is the architectural commitment that makes the halt-router / write-gate subspace question well-posed.

## Layout

```
transadapter/
  blocks/           workspace, specialists, mor_router, feedback
  baselines/        v_vanilla (matched-shape baseline)
  eval/             broadcast_events, codim_overlap, controller_id, circuits  (planned)
  model.py          TransAdapter top-level
  config.py         single source of truth for shapes + ablation toggles
configs/            YAML configs per scale     (planned)
scripts/            smoke_forward, train, eval entry points
tests/              unit tests
preregistration/    frozen analysis pipeline + locked hypotheses  (planned)
papers/             positive_outline.md + null_outline.md         (planned)
saes/               trained SAEs for mech-interp                  (planned)
```

## Ablation toggles wired today

- `enable_router_reads_workspace = False` — router reads only the stream (decouples halt subspace from `W`)
- `enable_feedback = False` — disables top-down feedback

Both are exercised in `tests/test_smoke.py`. The full ablation ladder (workspace-off, registers baseline, workspace+MoR without feedback, etc.) is planned.

## Hardware

Architecture code is CPU-runnable for forward-pass tests. A 10M × 500M-token training run as currently scoped exceeds a 4 GB GPU; scope or hardware will need to change before Phase 0 training begins.

## License

Private research repository. No license granted.

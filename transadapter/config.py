from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransAdapterConfig:
    """Hyperparameters for TransAdapter and matched baselines.

    All shape-determining fields drive both the model and the matched-FLOPs/params
    baseline counters. Keep this single source of truth so the
    ablation ladder stays comparable.
    """

    # tokenizer / sequence
    vocab_size: int = 50304
    max_seq_len: int = 512

    # residual stream
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_head: int = 64

    # workspace W (persistent across the entire forward pass)
    n_slots: int = 8
    d_workspace: int = 192
    n_specialists: int = 3
    top_k_write: int = 2
    workspace_alpha_init: float = 0.5

    # parallel specialists
    sliding_window: int = 64
    ssm_state_dim: int = 32

    # MoR router (reads [S_t ; attn(S_t, W)])
    max_steps: int = 4
    router_hidden: int = 64
    halt_temperature: float = 1.0
    halt_eps: float = 0.01
    enable_router_reads_workspace: bool = True
    """If False, ablates the Q1 coupling: router reads only the stream."""

    # top-down feedback (W -> previous layer)
    feedback_rank: int = 32
    feedback_warmup_frac: float = 0.05
    enable_feedback: bool = True

    # regularization / norm
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5

    # tied embedding
    tie_embeddings: bool = True

    # training-time constants used by the model (e.g. for warmup curves)
    train_total_tokens: int = 0
    """Set by the trainer; 0 disables warmup (gate = learned value immediately)."""

    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if self.n_specialists < 1:
            raise ValueError("n_specialists must be >= 1")
        if self.top_k_write > self.n_specialists * self.n_slots:
            raise ValueError("top_k_write exceeds total candidate (specialist x slot) pairs")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

"""nanoGPT-style decoder-only transformer in MLX.

Pre-LN blocks, rotary position embeddings, SwiGLU (or GELU) MLP, tied input/output
embeddings, no biases. Weights are kept in float32; ``compute_dtype`` casts weights
and activations for the matmuls (mixed precision), while norms, softmax and the
loss stay in float32.

``__call__(..., return_hidden=True)`` also returns the residual stream after the
embedding and after every block (used by the Phase 4 probes and CKA).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

# Named sizes. Parameter counts include the tied embedding (vocab ~1.25k); see README.
SIZES = {
    "S": dict(n_layers=4, d_model=192, n_heads=4),  # ~2.0M
    "M": dict(n_layers=6, d_model=320, n_heads=5),  # ~7.8M
    "L": dict(n_layers=6, d_model=512, n_heads=8),  # ~19.6M
    "XS": dict(n_layers=2, d_model=64, n_heads=2),  # tests / smoke only
}


@dataclass
class ModelConfig:
    vocab_size: int = 0  # set from the encoder's vocabulary
    size: str | None = "S"  # a key of SIZES, or None to use the explicit fields below
    n_layers: int = 4
    d_model: int = 192
    n_heads: int = 4
    mlp: str = "swiglu"  # swiglu | gelu
    rope_base: float = 10000.0
    max_len: int = 1024  # only used for sanity checks; RoPE has no learned table
    init_std: float = 0.02
    compute_dtype: str = "float32"  # float32 | bfloat16 | float16

    def resolved(self) -> "ModelConfig":
        d = asdict(self)
        if self.size is not None:
            d.update(SIZES[self.size])
        return ModelConfig(**d)

    @property
    def mlp_hidden(self) -> int:
        if self.mlp == "gelu":
            return 4 * self.d_model
        # SwiGLU: 2/3 * 4d rounded up to a multiple of 64 (keeps params ~= GELU MLP)
        h = int(8 * self.d_model / 3)
        return 64 * math.ceil(h / 64)


DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


class Linear(nn.Module):
    """Bias-free linear layer whose matmul runs in the compute dtype."""

    def __init__(self, d_in: int, d_out: int, dtype=mx.float32):
        super().__init__()
        self.weight = mx.zeros((d_out, d_in))
        self._dtype = dtype

    def __call__(self, x):
        return x @ self.weight.astype(self._dtype).T


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, dtype):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = Linear(cfg.d_model, 3 * cfg.d_model, dtype)
        self.out = Linear(cfg.d_model, cfg.d_model, dtype)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base)

    def __call__(self, x, mask):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, dtype):
        super().__init__()
        self.kind = cfg.mlp
        h = cfg.mlp_hidden
        if cfg.mlp == "swiglu":
            self.gate = Linear(cfg.d_model, h, dtype)
        self.up = Linear(cfg.d_model, h, dtype)
        self.down = Linear(h, cfg.d_model, dtype)

    def __call__(self, x):
        if self.kind == "swiglu":
            return self.down(nn.silu(self.gate(x)) * self.up(x))
        return self.down(nn.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, dtype):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = Attention(cfg, dtype)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg, dtype)
        self._dtype = dtype

    def __call__(self, x, mask):
        # x is the float32 residual stream; sublayers compute in the compute dtype
        x = x + self.attn(self.ln1(x).astype(self._dtype), mask).astype(mx.float32)
        x = x + self.mlp(self.ln2(x).astype(self._dtype)).astype(mx.float32)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg = cfg.resolved()
        assert cfg.vocab_size > 0, "vocab_size must be set"
        self.cfg = cfg
        dtype = DTYPES[cfg.compute_dtype]
        self._dtype = dtype
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg, dtype) for _ in range(cfg.n_layers)]
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.init_std
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)

        def init(path, p):
            if path.endswith("ln1.weight") or path.endswith("ln2.weight") or path.endswith("ln_f.weight"):
                return mx.ones_like(p)
            s = resid_std if (path.endswith("out.weight") or path.endswith("down.weight")) else std
            return mx.random.normal(p.shape) * s

        self.update(tree_map_with_path(init, self.parameters()))

    def __call__(self, tokens, mask="causal", return_hidden: bool = False):
        x = self.embed(tokens).astype(mx.float32)
        hidden = [x] if return_hidden else None
        for blk in self.blocks:
            x = blk(x, mask)
            if return_hidden:
                hidden.append(x)
        x = self.ln_f(x)
        logits = (x.astype(self._dtype) @ self.embed.weight.astype(self._dtype).T).astype(mx.float32)
        return (logits, hidden) if return_hidden else logits


def count_params(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def padding_mask(pad: mx.array, dtype=mx.float32) -> mx.array:
    """Additive causal mask that also hides padded keys. pad: (B, T) bool, True = padding."""
    T = pad.shape[1]
    causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    allowed = causal[None, None, :, :] & ~pad[:, None, None, :]
    return mx.where(allowed, mx.array(0.0, dtype), mx.array(-1e9, dtype))

"""Pure-NumPy inference for the exit transformer.

Mirrors `algo/diagnostics/exit_transformer/model.py` exactly. Loads weights
from `exit_model.npz` and exposes a single function `predict(seq, scalar)`
that returns forward-return predictions in bps.

The Pi runs only this module (plus numpy) — no torch dependency.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DEFAULT_NPZ = Path(__file__).resolve().parent.parent / "state" / "exit_transformer" / "exit_model.npz"


def _gelu(x):
    # exact GELU (matches torch nn.GELU)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _layernorm(x, weight, bias, eps=1e-5):
    # x: (..., D); normalize over last dim
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _linear(x, weight, bias):
    # weight: (out, in), x: (..., in)
    return x @ weight.T + bias


def _multihead_attention(x, in_w, in_b, out_w, out_b, n_heads):
    """Self-attention. x: (B, T, D). Mirrors torch nn.MultiheadAttention with
    in_proj_weight / in_proj_bias packed [Q, K, V]. batch_first=True.
    """
    B, T, D = x.shape
    head_dim = D // n_heads
    qkv = x @ in_w.T + in_b  # (B, T, 3D)
    q, k, v = np.split(qkv, 3, axis=-1)
    # Reshape to (B, n_heads, T, head_dim)
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale  # (B, h, T, T)
    # softmax
    scores -= scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    attn = exp / exp.sum(axis=-1, keepdims=True)
    out = np.matmul(attn, v)  # (B, h, T, head_dim)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ out_w.T + out_b


def _encoder_layer(x, w, layer_idx, n_heads):
    """One TransformerEncoderLayer in 'post-norm' (PyTorch default with
    norm_first=False). batch_first=True, activation=GELU."""
    p = f"encoder.layers.{layer_idx}."
    # self-attn
    attn = _multihead_attention(
        x,
        in_w=w[p + "self_attn.in_proj_weight"],
        in_b=w[p + "self_attn.in_proj_bias"],
        out_w=w[p + "self_attn.out_proj.weight"],
        out_b=w[p + "self_attn.out_proj.bias"],
        n_heads=n_heads,
    )
    x = _layernorm(x + attn, w[p + "norm1.weight"], w[p + "norm1.bias"])
    # FFN
    h = _linear(x, w[p + "linear1.weight"], w[p + "linear1.bias"])
    h = _gelu(h)
    h = _linear(h, w[p + "linear2.weight"], w[p + "linear2.bias"])
    x = _layernorm(x + h, w[p + "norm2.weight"], w[p + "norm2.bias"])
    return x


class ExitModelNumpy:
    """Loads model weights + norm stats from a .npz file."""

    def __init__(self, npz_path: Path | str = DEFAULT_NPZ):
        d = np.load(npz_path, allow_pickle=True)
        self.weights = {k: d[k] for k in d.files if not k.startswith("_")}
        self.seq_mean = d["_seq_mean"]
        self.seq_std = d["_seq_std"]
        self.sca_mean = d["_sca_mean"]
        self.sca_std = d["_sca_std"]
        self.y_mean = d["_y_mean"]
        self.y_std = d["_y_std"]
        self.config = json.loads(str(d["_config_json"][0]))
        self.n_heads = self.config["nhead"]
        self.num_layers = self.config["num_layers"]
        self.targets = self.config["targets"]
        self.seq_len = self.config["seq_len"]

    def predict(self, seq: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        """Returns forward-return predictions in bps.
        seq: (B, T, F_seq)  raw (will be z-scored using stored stats)
        scalar: (B, F_scalar)
        out: (B, n_targets) in bps (de-normalised)
        """
        if seq.ndim == 2:
            seq = seq[None, ...]
        if scalar.ndim == 1:
            scalar = scalar[None, ...]
        seq_n = (seq - self.seq_mean) / self.seq_std
        sca_n = (scalar - self.sca_mean) / self.sca_std
        return self._forward_normalised(seq_n.astype(np.float32),
                                        sca_n.astype(np.float32))

    def _forward_normalised(self, seq, scalar):
        w = self.weights
        # proj: (B, T, D)
        x = _linear(seq, w["proj.weight"], w["proj.bias"])
        # cls token
        B = x.shape[0]
        cls = np.broadcast_to(w["cls"], (B, 1, x.shape[-1])).copy()
        x = np.concatenate([cls, x], axis=1)
        # positional encoding
        x = x + w["pos.pe"][None, : x.shape[1]]
        # encoder layers
        for li in range(self.num_layers):
            x = _encoder_layer(x, w, li, self.n_heads)
        seq_pool = x[:, 0]  # cls token
        # scalar proj + GELU
        sc = _linear(scalar, w["scalar_proj.0.weight"], w["scalar_proj.0.bias"])
        sc = _gelu(sc)
        # head: linear -> GELU -> linear (note dropout was 0.0 at inference)
        z = np.concatenate([seq_pool, sc], axis=1)
        h = _linear(z, w["head.0.weight"], w["head.0.bias"])
        h = _gelu(h)
        out = _linear(h, w["head.3.weight"], w["head.3.bias"])
        # de-normalise
        return out * self.y_std + self.y_mean


if __name__ == "__main__":
    # Sanity check: compare with torch
    m = ExitModelNumpy()
    print(f"loaded model: layers={m.num_layers}  heads={m.n_heads}  targets={m.targets}")
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((1, m.seq_len, 3)).astype(np.float32)
    sca = rng.standard_normal((1, 10)).astype(np.float32)
    # Pre-undo norm so the np.predict redoes the same z-score
    seq_raw = seq * m.seq_std + m.seq_mean
    sca_raw = sca * m.sca_std + m.sca_mean
    np_pred = m.predict(seq_raw, sca_raw)
    print(f"numpy prediction: {np_pred[0]}")

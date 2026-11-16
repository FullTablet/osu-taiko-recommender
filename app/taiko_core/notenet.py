"""NumPy forward pass of the note-difficulty subnet (difficulty_nn ObjectDifficultySubnet).

No PyTorch needed: weights come from the expansion pack's note_net.npz (written
by builder/build_expansion_pack.py, checked there against the PyTorch model).

Input: normalized note features, (B, n, 38) float32 -- B mod combinations of the
same chart (same note count). Output: slope (B, n, 5) >= 0 and bias (B, n, 5).
The per-note confidence weight W_j is computed by the net but not returned
(the app doesn't use it).

The attention is causal with a fixed window: note i sees notes i-window+1 .. i,
exactly like the training code, so charts are processed one at a time without
padding.
"""
from __future__ import annotations

import json
import math

import numpy as np

_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorized."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * a)
    a1, a2, a3, a4, a5 = _ERF_A
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-a * a)
    return s * y


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU, as torch.nn.GELU()."""
    return 0.5 * x * (1.0 + _erf(x * (1.0 / math.sqrt(2.0))))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


class NoteNet:
    """Loaded from note_net.npz. Keys: torch state-dict names of object_subnet
    (prefix stripped), plus 'config_json' (window, num_heads, num_layers, skill_dim,
    aa_dim, object_feature_dim)."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.cfg = json.loads(str(z["config_json"]))
            self.p = {k: z[k].astype(np.float32) for k in z.files if k != "config_json"}
        self.window = int(self.cfg["window"])
        self.heads = int(self.cfg["num_heads"])
        self.layers = int(self.cfg["num_layers"])
        self.skill_dim = int(self.cfg["skill_dim"])
        self.feature_dim = int(self.cfg["object_feature_dim"])
        self._masks = self._base_masks(self.window)

    @staticmethod
    def _base_masks(w: int):
        i = np.arange(w)[:, None]
        k = np.arange(2 * w)[None, :]
        base = ((k >= w) & (k <= w + i)) | ((k < w) & (k >= i + 1))
        first = base.copy()
        first[:, :w] = False
        return base, first

    def _linear(self, x, name):
        return x @ self.p[name + ".weight"].T + self.p[name + ".bias"]

    def _attention(self, x, prefix):
        b, n, h = x.shape
        w = self.window
        nb = -(-n // w)
        pad = nb * w - n
        if pad:
            x = np.concatenate([x, np.zeros((b, pad, h), x.dtype)], axis=1)
        qkv = self._linear(x, prefix + ".qkv_proj")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(b, nb, w, h)
        k = k.reshape(b, nb, w, h)
        v = v.reshape(b, nb, w, h)
        zeros = np.zeros((b, 1, w, h), x.dtype)
        k_cat = np.concatenate([np.concatenate([zeros, k[:, :-1]], axis=1), k], axis=2)   # (b, nb, 2w, h)
        v_cat = np.concatenate([np.concatenate([zeros, v[:, :-1]], axis=1), v], axis=2)
        nh = self.heads
        dh = h // nh
        qh = q.reshape(b, nb, w, nh, dh).transpose(0, 1, 3, 2, 4)                       # (b, nb, nh, w, dh)
        kh = k_cat.reshape(b, nb, 2 * w, nh, dh).transpose(0, 1, 3, 2, 4)
        vh = v_cat.reshape(b, nb, 2 * w, nh, dh).transpose(0, 1, 3, 2, 4)
        scores = qh @ kh.transpose(0, 1, 2, 4, 3) / np.float32(math.sqrt(dh))           # (b, nb, nh, w, 2w)
        base, first = self._masks
        scores = np.where(base, scores, -np.inf)
        scores[:, :1] = np.where(first, scores[:, :1], -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(axis=-1, keepdims=True)
        out = (attn @ vh).transpose(0, 1, 3, 2, 4).reshape(b, nb * w, h)[:, :n]
        return self._linear(out, prefix + ".out_proj")

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """features (B, n, 38) normalized -> (slope (B, n, 5), bias (B, n, 5)), float32."""
        x = np.asarray(features, np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.shape[1] == 0:
            z = np.zeros((x.shape[0], 0, self.skill_dim), np.float32)
            return z, z.copy()
        p = self.p
        x = self._linear(x, "input_proj")
        for i in range(self.layers):
            pre = f"blocks.{i}"
            x = x + self._attention(layer_norm(x, p[pre + ".norm1.weight"], p[pre + ".norm1.bias"]), pre + ".attn")
            hdn = layer_norm(x, p[pre + ".norm2.weight"], p[pre + ".norm2.bias"])
            hdn = gelu(self._linear(hdn, pre + ".ffn.0"))
            x = x + self._linear(hdn, pre + ".ffn.3")
        x = layer_norm(x, p["final_norm.weight"], p["final_norm.bias"])
        out = self._linear(gelu(self._linear(x, "head.0")), "head.2")
        d = self.skill_dim
        slope = softplus(out[..., :d]).astype(np.float32)
        bias = out[..., d:2 * d].astype(np.float32)
        return slope, bias

"""Skill quantile tables: value <-> quantile per skill dimension.

Each dimension has K keypoints: values v_0 <= ... <= v_{K-1} at quantiles
q_i = i / (K-1) (q_0 = 0: worst observed, q_{K-1} = 1: best observed).
Between keypoints the mapping is linear. Outside [0, 1] it continues linearly
with a robust tail slope (the secant over the outer `tail` fraction, e.g. the
top 1% of plays), anchored at the end keypoint so both directions are
continuous at quantile 0 and 1:

    value(q) = v_0     + (q - 0) * slope_low     for q < 0
    value(q) = v_{K-1} + (q - 1) * slope_high    for q > 1

and quantile(v) is the exact inverse. Displayed percentiles may therefore be
below 0% or above 100%.
"""
from __future__ import annotations

import json

import numpy as np


class QuantileTable:
    def __init__(self, values: np.ndarray, slope_low: np.ndarray, slope_high: np.ndarray, meta: dict | None = None):
        self.values = np.asarray(values, np.float64)               # (D, K), non-decreasing per row
        self.D, self.K = self.values.shape
        self.q = np.linspace(0.0, 1.0, self.K)
        self.slope_low = np.maximum(np.asarray(slope_low, np.float64), 1e-12)
        self.slope_high = np.maximum(np.asarray(slope_high, np.float64), 1e-12)
        self.meta = meta or {}

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_samples(cls, samples: np.ndarray, keypoints: int = 1000, tail: float = 0.01) -> "QuantileTable":
        """samples (N, D) -> table with `keypoints` points per dimension; tail slopes are
        the secants over the bottom/top `tail` fraction."""
        samples = np.asarray(samples, np.float64)
        q = np.linspace(0.0, 1.0, keypoints)
        values = np.quantile(samples, q, axis=0).T                  # (D, K)
        values = np.maximum.accumulate(values, axis=1)              # guard against fp non-monotonicity
        lo = np.quantile(samples, tail, axis=0)
        hi = np.quantile(samples, 1.0 - tail, axis=0)
        slope_low = (lo - values[:, 0]) / tail
        slope_high = (values[:, -1] - hi) / tail
        return cls(values, slope_low, slope_high, {"tail": tail, "n_samples": int(len(samples))})

    def to_json(self) -> str:
        return json.dumps({"values": self.values.tolist(), "slope_low": self.slope_low.tolist(),
                           "slope_high": self.slope_high.tolist(), "meta": self.meta})

    @classmethod
    def from_json(cls, text: str) -> "QuantileTable":
        d = json.loads(text)
        return cls(np.array(d["values"]), np.array(d["slope_low"]), np.array(d["slope_high"]), d.get("meta"))

    # ---------------------------------------------------------------- queries
    def value(self, dim: int, q):
        """Direct query: value of quantile q (scalar or array) in dimension dim."""
        q = np.asarray(q, np.float64)
        v = self.values[dim]
        out = np.interp(q, self.q, v)
        out = np.where(q < 0, v[0] + q * self.slope_low[dim], out)
        out = np.where(q > 1, v[-1] + (q - 1.0) * self.slope_high[dim], out)
        return out if out.ndim else float(out)

    def quantile(self, dim: int, value):
        """Reverse query: quantile of `value` (scalar or array) in dimension dim."""
        x = np.asarray(value, np.float64)
        v = self.values[dim]
        out = _interp_monotone(x, v, self.q)
        out = np.where(x < v[0], (x - v[0]) / self.slope_low[dim], out)
        out = np.where(x > v[-1], 1.0 + (x - v[-1]) / self.slope_high[dim], out)
        return out if out.ndim else float(out)

    def values_at(self, quantiles) -> np.ndarray:
        """Skill vector whose dimension k sits at quantiles[k] (array of length D, or scalar for all).
        Vectorized over dimensions (the keypoints are evenly spaced in quantile)."""
        qs = np.broadcast_to(np.asarray(quantiles, np.float64), (self.D,))
        k = self.K - 1
        pos = np.clip(qs, 0.0, 1.0) * k
        i0 = np.minimum(pos.astype(np.int64), k - 1)
        f = pos - i0
        rows = np.arange(self.D)
        out = self.values[rows, i0] * (1.0 - f) + self.values[rows, i0 + 1] * f
        out = np.where(qs < 0, self.values[:, 0] + qs * self.slope_low, out)
        return np.where(qs > 1, self.values[:, -1] + (qs - 1.0) * self.slope_high, out)

    def quantiles_of(self, skill) -> np.ndarray:
        s = np.asarray(skill, np.float64)
        return np.array([self.quantile(k, s[k]) for k in range(self.D)])


def _interp_monotone(x, xp, fp):
    """np.interp for a non-decreasing xp that may contain flat runs (ties): a value
    equal to a tied run maps to the middle of the run's quantiles."""
    lo = np.interp(x, xp, fp)                      # numpy takes the last of equal xp
    rev = -xp[::-1]
    hi = np.interp(-x, rev, fp[::-1])              # ... and this the first
    return 0.5 * (lo + hi)

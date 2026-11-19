"""Per-note expected accuracy curve for the chart details window (NumPy only).

smooth():  Gaussian blur like Mathematica's GaussianFilter[data, r]: kernel radius r
           (2r + 1 taps), standard deviation r/2, normalized to sum 1; the ends are
           padded by repeating the first / last value ("Fixed" padding).
colors():  colour of each point from its distance to the mean in standard deviations:
           z <= -2 red, z = 0 yellow, z >= +2 blue, with a smooth hue transition
           (red -> orange -> yellow -> green -> cyan -> blue).
"""
from __future__ import annotations

import colorsys

import numpy as np

SMOOTH_RADIUS = 2
Z_LIMIT = 2.0


def gaussian_kernel(radius: int = SMOOTH_RADIUS, sigma: float | None = None) -> np.ndarray:
    sigma = radius / 2.0 if sigma is None else sigma
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2) if sigma > 0 else (x == 0).astype(np.float64)
    return k / k.sum()


def smooth(values, radius: int = SMOOTH_RADIUS, sigma: float | None = None) -> np.ndarray:
    v = np.asarray(values, np.float64)
    if len(v) == 0 or radius <= 0:
        return v.copy()
    padded = np.concatenate([np.full(radius, v[0]), v, np.full(radius, v[-1])])
    return np.convolve(padded, gaussian_kernel(radius, sigma), mode="valid")


def z_scores(curve, mean: float, sd: float) -> np.ndarray:
    c = np.asarray(curve, np.float64)
    if not sd > 0:
        return np.zeros_like(c)
    return (c - mean) / sd


def hue_for_z(z) -> np.ndarray:
    """Hue in degrees: -2 sd (or less) red 0, mean yellow 60, +2 sd (or more) blue 220."""
    t = (np.clip(np.asarray(z, np.float64), -Z_LIMIT, Z_LIMIT) + Z_LIMIT) / (2 * Z_LIMIT)    # 0..1
    return np.where(t <= 0.5, 120.0 * t, 60.0 + 320.0 * (t - 0.5))


def colors(z, saturation: float = 0.95, value: float = 0.85) -> list[tuple[int, int, int]]:
    out = []
    for h in hue_for_z(z):
        r, g, b = colorsys.hsv_to_rgb(float(h) / 360.0, saturation, value)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return out


def curve_stats(per_note) -> dict:
    """mean = the chart's expected accuracy (mean of the per-note values);
    sd = spread of the smoothed curve, which is what the colours describe."""
    p = np.asarray(per_note, np.float64)
    sm = smooth(p)
    mean = float(p.mean()) if len(p) else float("nan")
    sd = float(sm.std()) if len(sm) > 1 else 0.0
    return dict(smoothed=sm, mean=mean, sd=sd, z=z_scores(sm, mean, sd))

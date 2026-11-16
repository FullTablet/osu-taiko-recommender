"""Skill maths: expected accuracy, skill estimation from plays, and skill
quantiles needed for a target accuracy along a direction.

Model (same as training, without the per-note confidence weight W_j):
    x_j  = sum_k slope_jk * (s_k - bias_jk)          note logit for skill s
    p_j  = sigmoid(x_j)                               expected "accuracy" of note j
    chart expected accuracy = mean_j p_j

Skill estimation maximizes the recency-weighted Bernoulli log-likelihood
    sum_plays w_p sum_notes [ S_p x_pj - softplus(x_pj) ]   (S_p = play accuracy in [0, 1])
with w = 1, 0.99, 0.99^2, ... from the most recent play backwards (at most 1000
plays). It is concave in s, so Newton's method converges in a few steps. A very
weak Gaussian prior around a reference skill (default: the previous estimate)
keeps the estimate finite when every play is 100% (or 0%); with real play counts
it has no visible effect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RECENCY_DECAY = 0.99
MAX_PLAYS = 1000
PRIOR_PRECISION = 1.0       # in skill units^-2; one note at p=0.5 with slope 1 has curvature 0.25


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def note_logits(slope: np.ndarray, bias: np.ndarray, skill) -> np.ndarray:
    s = np.asarray(skill, np.float64)
    slope = np.asarray(slope, np.float64)
    return slope @ s - (slope * np.asarray(bias, np.float64)).sum(axis=-1)


def note_probabilities(slope, bias, skill) -> np.ndarray:
    """Expected per-note hit probability (for a future 'choke point' graph)."""
    return sigmoid(note_logits(slope, bias, skill))


def expected_accuracy(slope, bias, skill) -> float:
    if len(slope) == 0:
        return float("nan")
    return float(note_probabilities(slope, bias, skill).mean())


# --------------------------------------------------------------------------- estimation
@dataclass
class PlayData:
    slope: np.ndarray        # (n, 5)
    bias: np.ndarray         # (n, 5)
    accuracy: float          # 0..1


def recency_weights(n_plays: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Weights for plays ordered OLDEST first: the last (most recent) gets 1."""
    return decay ** np.arange(n_plays - 1, -1, -1, dtype=np.float64)


def estimate_skill(plays: list[PlayData], prior_mean, prior_precision: float = PRIOR_PRECISION,
                   decay: float = RECENCY_DECAY, max_plays: int = MAX_PLAYS, max_iter: int = 100,
                   tol: float = 1e-9, progress=None) -> tuple[np.ndarray, dict]:
    """plays: oldest first. Returns (skill vector, info)."""
    prior_mean = np.asarray(prior_mean, np.float64)
    plays = plays[-max_plays:]
    if not plays:
        return prior_mean.copy(), {"iterations": 0, "notes": 0, "plays": 0}
    w_play = recency_weights(len(plays), decay)
    a = np.concatenate([np.asarray(p.slope, np.float64) for p in plays])                  # (N, D)
    c = np.concatenate([(np.asarray(p.slope, np.float64) * np.asarray(p.bias, np.float64)).sum(1)
                        for p in plays])                                                   # (N,)
    counts = [len(p.slope) for p in plays]
    target = np.repeat([float(p.accuracy) for p in plays], counts)
    w = np.repeat(w_play, counts)
    d = a.shape[1]

    def objective(s):
        x = a @ s - c
        ll = (w * (target * x - np.logaddexp(0.0, x))).sum()
        return ll - 0.5 * prior_precision * ((s - prior_mean) ** 2).sum()

    s = prior_mean.copy()
    f = objective(s)
    it = 0
    for it in range(1, max_iter + 1):
        x = a @ s - c
        p = sigmoid(x)
        g = a.T @ (w * (target - p)) - prior_precision * (s - prior_mean)
        h = (a * (w * p * (1 - p))[:, None]).T @ a + prior_precision * np.eye(d)        # = -Hessian
        step = np.linalg.solve(h, g)
        t = 1.0
        while True:                                                                      # backtracking
            s_new = s + t * step
            f_new = objective(s_new)
            if f_new >= f - 1e-12 or t < 1e-8:
                break
            t *= 0.5
        converged = abs(f_new - f) <= tol * (1 + abs(f)) and np.abs(t * step).max() < 1e-7
        s, f = s_new, f_new
        if progress:
            progress(min(it / 20, 0.99))
        if converged:
            break
    return s, {"iterations": it, "notes": int(len(target)), "plays": len(plays), "log_likelihood": float(f)}


# --------------------------------------------------------------------------- pp rank -> quantile
RANK_POINTS = [(1.0, 1.0), (10_000.0, 0.3), (100_000.0, 0.01)]


def rank_to_quantile(rank: float) -> float:
    """Representative skill quantile for a global taiko pp rank: piecewise linear in
    log10(rank) through (1, 1.0), (10,000, 0.3), (100,000, 0.01); continued linearly
    beyond rank 100,000 (may go below 0)."""
    r = np.log10(max(float(rank), 1.0))
    xs = [np.log10(p[0]) for p in RANK_POINTS]
    ys = [p[1] for p in RANK_POINTS]
    if r <= xs[-1]:
        return float(np.interp(r, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + (r - xs[-1]) * slope)


# --------------------------------------------------------------------------- direction search
def quantiles_for_accuracy(slope, bias, table, direction, target: float,
                           t_limit: float = 50.0, iters: int = 80) -> np.ndarray | None:
    """Skill quantiles q = t * direction (t scalar) at which the chart's expected
    accuracy equals `target` (0..1). Starts from quantile 0 in every dimension and
    moves along `direction` (e.g. [1,1,1,1,1] balanced, [2,1,1,1,1] strong in dim 1).
    Returns the quantile vector, or None if unreachable within |t| <= t_limit."""
    direction = np.asarray(direction, np.float64)
    if not 0.0 < target < 1.0 or np.any(direction < 0) or not np.any(direction > 0):
        raise ValueError("target must be in (0, 1) and direction non-negative and non-zero")

    def acc(t):
        return expected_accuracy(slope, bias, table.values_at(t * direction))

    lo, hi = 0.0, 1.0
    a_lo, a_hi = acc(lo), acc(hi)
    while a_lo > target:                       # need quantiles below 0
        hi, a_hi = lo, a_lo
        lo = -1.0 if lo == 0 else lo * 2
        if lo < -t_limit:
            return None
        a_lo = acc(lo)
    while a_hi < target:                       # need quantiles above 1
        lo, a_lo = hi, a_hi
        hi *= 2
        if hi > t_limit:
            return None
        a_hi = acc(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if acc(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi) * direction

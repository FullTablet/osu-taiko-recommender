"""Chart recommendation: fit error and skill priorities (accuracies and quantiles in 0..1).

For a candidate chart (one mod combination) and the player's current skill:

  ha        expected accuracy of the player on the chart
  hapower   the quantile q at which a balanced skill vector (every dimension at
            quantile q) gives the same expected accuracy ha
  fiterror  softplus(hapower) + 25e12 * (distance of ha outside [target - 0.01, target + 0.01])^2

With a skill priority (dimension k):
  a(x) = softplus(x) + x/1e4            quantile of the ordinary dimensions on the path
  b(x) = 2 softplus(x) + x/1e4          quantile of the prioritized dimension
  x_c: expected accuracy = target with every dimension at a(x)
  x_p: expected accuracy = target with dimension k at b(x), the others at a(x)
  chartpower = softplus(x_c), prioritypower = softplus(x_p)
  priority_k = chartpower / prioritypower - 1          (>= 0: see below)
  fiterror /= priority_k^5

Why softplus (not a) in the powers: b(x) > a(x) for every x, so x_p <= x_c, and
softplus is positive and increasing, hence priority >= 0 always. With a(x)
(which is negative for x < -7.2315) very easy charts could get a negative
priority and jump to the top of the ranking. (Decision of 2026-11-16.)

Safeguards against rounding:
  * priority is computed as expm1(log softplus(x_c) - log softplus(x_p)), stable when
    the two are nearly equal and when softplus underflows (x << 0, where log softplus(x) ~ x);
  * priorities below PRIORITY_FLOOR (incl. tiny negatives from root-finding noise and
    exact zeros) are set to PRIORITY_FLOOR, so the division stays finite and such charts
    sink to the bottom;
  * roots are bracketed (expanding the bracket as needed) and refined with the Illinois
    method plus bisection fallback; a target that can't be bracketed gives the bracket
    end, not an error. The paths a(x), b(x) keep the x/1e4 term, so the search does go
    below quantile 0 for very easy charts (only the powers drop it);
  * hapower solves for ha clamped to [1e-12, 1 - 1e-12] (see hapower()).
"""
from __future__ import annotations

import math

import numpy as np

WINDOW = 0.01
PENALTY = 25e12
PRIORITY_FLOOR = 1e-9
X_TOL = 1e-11
DIMENSIONS = ["α", "β", "γ", "δ", "ε"]
DIMENSION_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon"]


def softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


def log_softplus(x: float) -> float:
    """log(softplus(x)), accurate for very negative x (softplus ~ exp(x))."""
    if x < -30.0:
        return x - 0.5 * math.exp(x)          # log(log1p(e^x)) = x - e^x/2 + O(e^2x)
    return math.log(softplus(x))


def path_a(x: float) -> float:
    return softplus(x) + x / 1e4


def path_b(x: float) -> float:
    return 2.0 * softplus(x) + x / 1e4


class ChartModel:
    """Fast expected accuracy for one chart variant as a function of skill quantiles."""

    def __init__(self, slope: np.ndarray, bias: np.ndarray, table):
        self.slope = np.asarray(slope, np.float64)
        self.c = (self.slope * np.asarray(bias, np.float64)).sum(axis=1)
        self.table = table
        self.n = len(self.slope)

    def accuracy_at_skill(self, skill) -> float:
        x = self.slope @ np.asarray(skill, np.float64) - self.c
        return float((0.5 * (1.0 + np.tanh(0.5 * x))).mean())

    def accuracy_at_quantiles(self, q) -> float:
        return self.accuracy_at_skill(self.table.values_at(q))


def solve_increasing(g, lo: float, hi: float, lo_limit: float, hi_limit: float, xtol: float = X_TOL,
                     max_iter: int = 200) -> float:
    """Root of an increasing function g (g(lo) < 0 < g(hi) after bracket expansion).
    Returns the bracket end if the root lies beyond the limits."""
    glo, ghi = g(lo), g(hi)
    step = max(hi - lo, 1.0)
    while glo > 0:                                    # expand downwards
        if lo <= lo_limit:
            return lo_limit
        hi, ghi = lo, glo
        lo = max(lo - step, lo_limit)
        step *= 2
        glo = g(lo)
    step = max(hi - lo, 1.0)
    while ghi < 0:                                    # expand upwards
        if hi >= hi_limit:
            return hi_limit
        lo, glo = hi, ghi
        hi = min(hi + step, hi_limit)
        step *= 2
        ghi = g(hi)
    if glo == 0:
        return lo
    if ghi == 0:
        return hi
    side = 0
    width = hi - lo
    bisect_next = False
    for _ in range(max_iter):
        if hi - lo <= xtol * (1.0 + abs(lo) + abs(hi)):
            break
        if bisect_next or ghi == glo:
            x = 0.5 * (lo + hi)
        else:
            x = (lo * ghi - hi * glo) / (ghi - glo)          # regula falsi (Illinois)
            if not lo < x < hi:
                x = 0.5 * (lo + hi)
        gx = g(x)
        if gx == 0:
            return x
        if gx < 0:
            lo, glo = x, gx
            if side == -1:
                ghi *= 0.5
            side = -1
        else:
            hi, ghi = x, gx
            if side == +1:
                glo *= 0.5
            side = +1
        new_width = hi - lo
        bisect_next = new_width > 0.7 * width
        width = new_width
    return 0.5 * (lo + hi)


HA_CLAMP = 1e-12


def hapower(cm: ChartModel, ha: float) -> float:
    """Balanced quantile giving expected accuracy ha. ha is clamped to
    [1e-12, 1 - 1e-12]: an expected accuracy that rounds to exactly 0 or 1 has no finite
    root, and solving for the clamped value gives a well-defined quantile that keeps
    decreasing as the chart gets easier (instead of wherever rounding first reaches 1)."""
    h = min(max(float(ha), HA_CLAMP), 1.0 - HA_CLAMP)
    return solve_increasing(lambda q: cm.accuracy_at_quantiles(np.full(5, q)) - h,
                            -1.0, 2.0, -1e4, 1e4)


def solve_x(cm: ChartModel, target: float, priority_dim: int | None = None) -> float:
    """x such that the path quantiles (a(x) everywhere, b(x) in priority_dim) reach target."""
    def g(x):
        q = np.full(5, path_a(x))
        if priority_dim is not None:
            q[priority_dim] = path_b(x)
        return cm.accuracy_at_quantiles(q) - target

    return solve_increasing(g, -5.0, 5.0, -1e8, 1e4)


def priority_from_x(x_c: float, x_p: float) -> float:
    """chartpower / prioritypower - 1 with powers = softplus(x); floored (see module doc)."""
    p = math.expm1(log_softplus(x_c) - log_softplus(x_p))
    if not math.isfinite(p) or p < PRIORITY_FLOOR:
        return PRIORITY_FLOOR
    return p


def window_penalty(ha: float, target: float) -> float:
    if ha > target + WINDOW:
        return PENALTY * (ha - target - WINDOW) ** 2
    if ha < target - WINDOW:
        return PENALTY * (ha - target + WINDOW) ** 2
    return 0.0


def evaluate(cm: ChartModel, skill, target: float, priority_dim: int | None = None) -> dict:
    """Fit error of one chart variant (plus what's needed to reuse later)."""
    ha = cm.accuracy_at_skill(skill)
    hp = hapower(cm, ha)
    fit = softplus(hp) + window_penalty(ha, target)
    out = dict(ha=ha, hapower=hp, in_window=abs(ha - target) <= WINDOW, priorities={}, x_c=None)
    if priority_dim is not None:
        out["x_c"] = solve_x(cm, target)
        pr = priority_from_x(out["x_c"], solve_x(cm, target, priority_dim))
        out["priorities"][priority_dim] = pr
        fit = fit / pr ** 5
    out["fiterror"] = fit
    return out


def all_priorities(cm: ChartModel, target: float, known: dict | None = None, x_c: float | None = None) -> list[float]:
    """Priority of every dimension (reusing x_c and any priority already computed)."""
    known = dict(known or {})
    if x_c is None:
        x_c = solve_x(cm, target)
    return [known[k] if k in known else priority_from_x(x_c, solve_x(cm, target, k)) for k in range(5)]

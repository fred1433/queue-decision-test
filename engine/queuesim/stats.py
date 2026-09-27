"""Small, dependency-light statistics used by the engine and the power calendar."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

Z95 = 1.959963984540054


def wilson(k: float, n: float, z: float = Z95):
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mh_risk_ratio(a, n1, c, n0):
    """Mantel-Haenszel risk ratio across strata, Greenland-Robins 95% interval.

    a, c: events in arm 1 and arm 0 per stratum; n1, n0: arm sizes per stratum.
    Strata where one arm is empty carry no information and are dropped.
    """
    a, n1, c, n0 = (np.asarray(x, float) for x in (a, n1, c, n0))
    keep = (n1 > 0) & (n0 > 0)
    a, n1, c, n0 = a[keep], n1[keep], c[keep], n0[keep]
    N = n1 + n0
    R = np.sum(a * n0 / N)
    S = np.sum(c * n1 / N)
    if R <= 0 or S <= 0:
        return dict(rr=float("nan"), lo=float("nan"), hi=float("nan"), strata=int(keep.sum()),
                    events1=float(a.sum()), events0=float(c.sum()))
    rr = R / S
    v = np.sum((n1 * n0 * (a + c) - a * c * N) / N ** 2) / (R * S)
    se = math.sqrt(max(v, 0.0))
    return dict(rr=float(rr), lo=float(rr * math.exp(-Z95 * se)), hi=float(rr * math.exp(Z95 * se)),
                strata=int(len(N)), events1=float(a.sum()), events0=float(c.sum()))


def n_per_arm_props(p0: float, rel: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Units per arm to detect p0 -> p0*(1+rel), two-sided test."""
    p1 = min(p0 * (1 + rel), 0.999999)
    if p0 <= 0 or p1 == p0:
        return float("inf")
    za = st.norm.ppf(1 - alpha / 2)
    zb = st.norm.ppf(power)
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2


def n_per_arm_mean(mu: float, var: float, rel: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Units per arm to detect a relative change `rel` in a mean with per-unit variance `var`."""
    if mu <= 0:
        return float("inf")
    za = st.norm.ppf(1 - alpha / 2)
    zb = st.norm.ppf(power)
    return (za + zb) ** 2 * 2 * var / (rel * mu) ** 2


def verdict(lo: float, hi: float, null: float = 1.0) -> str:
    """'act' when the interval clears the null on the favourable side, 'reject' on the other,
    'undecided' when it straddles it (or is undefined)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undecided"
    if lo > null:
        return "above"
    if hi < null:
        return "below"
    return "undecided"

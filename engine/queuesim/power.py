"""A planner conditional on its assumptions, never a forecast.

Weeks of accrual for a fixed-horizon, two-sided test at 5% with 80% power, comparing two independent
proportions, when `leads_per_week` eligible, independent leads are randomized with `share` in the
treated arm. Outcome maturation comes on top. Clustering, restricted eligibility or several
comparisons change the design. The page runs the same formula in the browser (web/src/lib/planner.ts;
tests/test_power.py checks that both agree).
"""
from __future__ import annotations

import math

from scipy import stats as st

Z = st.norm.ppf(0.975) + st.norm.ppf(0.80)


def weeks_needed(p0: float, rel: float, leads_per_week: float, share: float = 0.5) -> float:
    p1 = min(p0 * (1 + rel), 0.999999)
    d = p1 - p0
    if p0 <= 0 or d == 0 or leads_per_week <= 0 or not 0 < share < 1:
        return math.inf
    # total leads N with n1 = share*N, n0 = (1-share)*N
    n_total = Z ** 2 * (p1 * (1 - p1) / share + p0 * (1 - p0) / (1 - share)) / d ** 2
    return n_total / leads_per_week


def scale_ratio_table(leads_per_week: float = 3200, collected: float = 3) -> list[dict]:
    """The assumed scale's ratio (3 collected per 3,200 leads), read as if it were a mature per-lead probability."""
    p0 = collected / leads_per_week
    return [dict(relative=r, p1=p0 * (1 + r), weeks=weeks_needed(p0, r, leads_per_week),
                 total_leads=weeks_needed(p0, r, leads_per_week) * leads_per_week)
            for r in (0.20, 0.50, 1.00)]

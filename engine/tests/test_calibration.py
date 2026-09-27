"""The synthetic operation lands near its assumed scale, and the collected count is NOT fixed."""
import numpy as np
from queuesim.simulate import simulate, weekly_totals


def test_scale_near_assumed_week():
    runs = [weekly_totals(simulate(4, s)[0]) for s in range(300, 304)]
    m = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
    assert abs(m["leads"] - 3200) / 3200 < 0.03
    assert abs(m["quotes"] - 1100) / 1100 < 0.07
    assert 32 <= m["issued"] <= 48
    assert 1.5 <= m["collected"] <= 5
    assert abs(m["ad_spend_mxn"] - 164000) < 1
    assert abs(m["call_center_cost_mxn"] - 234000) / 234000 < 0.02


def test_collected_varies_between_draws():
    logs, _ = simulate(8, 11)
    p = logs["policies"]
    wk = (p.issued_min // (7 * 1440)).astype(int)
    per_week = p[p.first_payment_collected].groupby(wk[p.first_payment_collected]).size()
    counts = [per_week.get(w, 0) for w in range(8)]
    assert len(set(counts)) > 1

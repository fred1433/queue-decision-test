"""Decision rules on hand-made logs, where the right answer is known without any simulation."""
import numpy as np
import pandas as pd
from queuesim.decide import decide_ownership, decide_script
from queuesim.stats import mh_risk_ratio, wilson


def make_logs(n, p_owner, p_current, kind="lead_random", seed=0, q=0.6):
    rng = np.random.default_rng(seed)
    owner = rng.random(n) < 0.5
    week = rng.integers(0, 8, n)
    if kind == "switchback":
        owner = np.isin(week, [0, 3, 5, 6])
    if kind == "from_week":
        owner = week >= 4
    reached = rng.random(n) < np.where(owner, p_owner, p_current)
    quoted = reached & (rng.random(n) < q)
    leads = pd.DataFrame(dict(lead_id=np.arange(n), week=week, arrival_min=week * 10080 + 600,
                              source=rng.choice(["meta", "google"], n), single_owner=owner, new_script=owner))
    rows = []
    for i in range(n):
        t = leads.arrival_min[i] + 5
        rows.append((i, t, 0, 10, 1, "A", "A01", 3.0, "A", "no_answer", 1.4))
        if reached[i]:
            rows.append((i, t + 200, 0, 13, 2, "B", "B01", 3.0, "A", "quote" if quoted[i] else "contact", 6.0))
    rows.append((0, 10 * 10080, 0, 10, 3, "A", "A01", 3.0, "A", "no_answer", 1.4))  # "now": two weeks after the last lead
    att = pd.DataFrame(rows, columns=["lead_id", "t_min", "day", "hour", "attempt_no", "handler", "agent_id",
                                      "agent_tenure_months", "script", "outcome", "handle_min"])
    pol = pd.DataFrame(columns=["lead_id", "handler", "issued_min", "premium_annual_mxn", "payment_on_call",
                                "first_payment_collected"])
    design = dict(ownership=dict(kind=kind), script=dict(kind="lead_random"))
    return dict(leads=leads, attempts=att, policies=pol, staffing=None, spend=None, design=design, meta=dict(weeks=8))


def test_history_only_is_test_only():
    logs = make_logs(4000, 0.5, 0.5)
    logs["design"]["ownership"] = dict(kind="off")
    assert decide_ownership(logs)["status"] == "test only"


def test_before_after_is_inadequate():
    assert decide_ownership(make_logs(4000, 0.6, 0.5, "from_week"))["status"] == "comparison inadequate"


def test_clear_gain_rolls_out():
    assert decide_ownership(make_logs(20000, 0.60, 0.50))["status"] == "roll out with review"


def test_clear_loss_keeps_current():
    assert decide_ownership(make_logs(20000, 0.45, 0.55))["status"] == "keep current"


def test_no_difference_is_not_yet():
    assert decide_ownership(make_logs(4000, 0.5, 0.5, seed=5))["status"] == "not yet"


def test_switchback_reads_weeks():
    d = decide_ownership(make_logs(40000, 0.62, 0.50, "switchback"))
    assert d["estimates"]["contacted"]["periods"] == [4, 4]
    assert d["status"] == "roll out with review"


def test_script_needs_collections_not_quotes():
    # more quotes, no collections recorded at all: never rolls out on quotes alone
    d = decide_script(make_logs(20000, 0.60, 0.50, q=0.6))
    assert d["estimates"]["quoted"]["lo"] > 1
    assert d["status"] != "roll out with review"
    assert d["naive"] == "roll out with review"


def test_mh_single_stratum_equals_plain_ratio():
    r = mh_risk_ratio([30], [100], [20], [100])
    assert abs(r["rr"] - 1.5) < 1e-12 and r["lo"] < 1.5 < r["hi"]


def test_wilson():
    p, lo, hi = wilson(3, 40)
    assert abs(p - 0.075) < 1e-12 and 0.02 < lo < 0.03 and 0.19 < hi < 0.21


def test_immature_leads_are_not_read():
    logs = make_logs(4000, 0.6, 0.5)
    logs["attempts"] = logs["attempts"][logs["attempts"].t_min < 9 * 10080]
    d = decide_ownership(logs)
    n = d["estimates"]["contacted"]["events1"] + d["estimates"]["contacted"]["events0"]
    assert n < 0.85 * (logs["attempts"].lead_id.nunique())

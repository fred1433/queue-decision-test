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
    assert decide_ownership(make_logs(20000, 0.60, 0.50))["status"] == "proceed to confirmation"


def test_clear_loss_keeps_current():
    assert decide_ownership(make_logs(20000, 0.45, 0.55))["status"] == "keep current"


def test_no_difference_is_not_yet():
    assert decide_ownership(make_logs(4000, 0.5, 0.5, seed=5))["status"] == "not yet"


def test_switchback_reads_weeks():
    d = decide_ownership(make_logs(40000, 0.62, 0.50, "switchback"))
    assert d["estimates"]["contacted"]["periods"] == [4, 4]
    assert d["status"] == "proceed to confirmation"


def test_script_needs_collections_not_quotes():
    # more quotes, no collections recorded at all: never rolls out on quotes alone
    d = decide_script(make_logs(20000, 0.60, 0.50, q=0.6))
    assert d["estimates"]["quoted"]["lo"] > 1
    assert d["status"] != "proceed to confirmation"
    assert d["naive"] == "roll out"


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


def test_collection_guardrail_holds_a_rollout():
    logs = make_logs(20000, 0.60, 0.50)
    L = logs["leads"]
    owners = L.lead_id[L.single_owner].to_numpy()[:40]
    others = L.lead_id[~L.single_owner].to_numpy()[:200]
    ids = list(owners) + list(others)
    logs["policies"] = pd.DataFrame(dict(lead_id=ids, handler="A", issued_min=0, premium_annual_mxn=9000.0,
                                         payment_on_call=False, first_payment_collected=True))
    d = decide_ownership(logs)
    assert d["estimates"]["contacted"]["lo"] > 1
    assert d["guardrail"] == "collections" and d["status"] == "not yet"


def test_sample_ratio_mismatch_is_inadequate():
    logs = make_logs(20000, 0.60, 0.50)
    L = logs["leads"]
    drop = L.index[L.single_owner][:1500]
    logs["leads"] = L.drop(drop).reset_index(drop=True)
    logs["design"]["ownership"] = dict(kind="lead_random", share=0.5)
    d = decide_ownership(logs)
    assert d["status"] == "comparison inadequate" and d["sample_ratio_p"] < 0.001


def _with_collections(logs, owner_events, current_events):
    L = logs["leads"]
    q = logs["attempts"][logs["attempts"].outcome == "quote"].lead_id
    own = [i for i in q if L.single_owner[i]][:owner_events]
    cur = [i for i in q if not L.single_owner[i]][:current_events]
    ids = own + cur
    arr = L.set_index("lead_id").arrival_min
    logs["policies"] = pd.DataFrame(dict(lead_id=ids, handler="A", issued_min=[int(arr[i]) + 1000 for i in ids],
                                         premium_annual_mxn=9000.0, payment_on_call=False,
                                         first_payment_collected=True))
    return logs


def test_zero_collections_in_owner_arm_trigger_the_guardrail():
    for cur in (10, 21, 200):
        d = decide_ownership(_with_collections(make_logs(20000, 0.60, 0.50), 0, cur))
        assert d["estimates"]["contacted"]["lo"] > 1
        assert d["guardrail"] == "collections" and d["status"] == "not yet", cur


def test_zero_collections_in_both_arms_is_explicitly_unresolved():
    d = decide_ownership(_with_collections(make_logs(20000, 0.60, 0.50), 0, 0))
    assert d["estimates"]["collected"]["resolved"] is False
    assert d["guardrail"] == "collections unresolved"
    assert "unresolved" in d["why"]


def test_zero_collections_in_current_arm_only_does_not_block():
    d = decide_ownership(_with_collections(make_logs(20000, 0.60, 0.50), 12, 0))
    assert d["estimates"]["collected"]["resolved"] and d["guardrail"] is None
    assert d["status"] == "proceed to confirmation"


def test_contacts_after_the_14_day_horizon_do_not_count():
    logs = make_logs(20000, 0.60, 0.50)
    a = logs["attempts"]
    L = logs["leads"].set_index("lead_id")
    late = a.outcome.isin(["contact", "quote"]) & L.single_owner.reindex(a.lead_id).to_numpy()
    a.loc[late, "t_min"] = a.loc[late, "t_min"] + 20 * 1440   # owner-arm contacts now land on day 20
    logs["meta"]["extracted_at_min"] = int(a.t_min.max())
    d = decide_ownership(logs)
    assert d["estimates"]["contacted"]["hi"] < 1
    assert d["status"] == "keep current"

"""The decision code. It reads operational logs and the logged design of any experiment, nothing else.

It never imports the simulator or its truth (tests/test_isolation.py checks the imports).

One decision: should each lead have a single next-contact owner across the three centers, instead of
the current process where any center may call it back? Plus one control case with the same
machinery: a new closing script, where the fast metric and the final outcome can disagree.

Every function returns a status among:
  "roll out with review"   the evidence clears the bar the decision record set in advance
  "keep current"           the evidence says the change hurts
  "not yet"                a valid comparison, not enough of it
  "test only"              no valid comparison exists in these logs: history cannot identify the effect
  "comparison inadequate"  a comparison exists but cannot carry the conclusion
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import assumptions as A
from .stats import mh_risk_ratio, wilson

AGE_BINS = [-1, 5, 30, 120, 720, 1e12]
AGE_LABELS = ["0-5m", "5-30m", "30m-2h", "2-12h", "12h+"]
NAN = float("nan")
STATUSES = ["roll out with review", "keep current", "not yet", "test only", "comparison inadequate"]


def _prep(logs):
    leads = logs["leads"].set_index("lead_id")
    a = logs["attempts"].copy()
    a["source"] = leads.source.reindex(a.lead_id).to_numpy()
    a["age_min"] = a.t_min - leads.arrival_min.reindex(a.lead_id).to_numpy()
    a["age_bin"] = pd.cut(a.age_min, AGE_BINS, labels=AGE_LABELS).astype(str)
    a["human"] = a.handler != "AI"
    a["contact"] = a.outcome.isin(["contact", "quote"])
    a["quote"] = a.outcome == "quote"
    a["att_bin"] = np.minimum(a.attempt_no, 6)
    return a


def lead_table(logs, a=None):
    """One row per lead: arm, first-dial outcome (before any overlap can happen), and outcomes."""
    a = _prep(logs) if a is None else a
    L = logs["leads"].set_index("lead_id").copy()
    g = a.groupby("lead_id")
    L["contacted"] = g.contact.any().reindex(L.index, fill_value=False)
    L["quoted"] = g.quote.any().reindex(L.index, fill_value=False)
    first = a.sort_values(["lead_id", "t_min"]).groupby("lead_id").head(1).set_index("lead_id")
    L["first_answered"] = first.contact.reindex(L.index, fill_value=False)
    L["dials"] = g.size().reindex(L.index, fill_value=0)
    L["agent_min"] = a[a.human].groupby("lead_id").handle_min.sum().reindex(L.index, fill_value=0.0)
    p = logs["policies"].set_index("lead_id")
    L["issued"] = L.index.isin(p.index)
    L["collected"] = p.first_payment_collected.reindex(L.index, fill_value=False).astype(bool)
    return L.reset_index()


def _rr(L, arm, outcome, strata):
    g = L.groupby(strata + [arm])[outcome].agg(["sum", "count"]).unstack(arm, fill_value=0)
    if True not in g["sum"].columns or False not in g["sum"].columns:
        return dict(rr=NAN, lo=NAN, hi=NAN, strata=0, events1=0.0, events0=0.0)
    return mh_risk_ratio(g["sum"][True], g["count"][True], g["sum"][False], g["count"][False])


# ----------------------------------------------------------------------------- what history shows

def overlaps(logs, a=None) -> dict:
    """The pattern every world shares: second dials by another center within minutes, and how
    the lead answers afterwards. Descriptive only."""
    a = _prep(logs) if a is None else a
    W = logs["meta"]["weeks"]
    hum = a[a.human].sort_values(["lead_id", "t_min"]).copy()
    prev_c = hum.groupby("lead_id").handler.shift()
    prev_t = hum.groupby("lead_id").t_min.shift()
    hum["overlap"] = prev_c.notna() & (prev_c != hum.handler) & ((hum.t_min - prev_t) <= A.COLLISION_WINDOW_MIN)
    cum = hum.groupby("lead_id").overlap.cumsum()
    hum["after"] = (cum - hum.overlap.astype(int)) > 0
    later = hum[~hum.overlap & (hum.attempt_no >= 2)]
    rr_after = _rr(later, "after", "contact", ["att_bin", "age_bin", "source"])
    second = hum[hum.overlap]
    return dict(
        leads_with_overlap_share=float(hum.loc[hum.overlap, "lead_id"].nunique() / len(logs["leads"])),
        overlaps_per_week=float(hum.overlap.sum() / W),
        second_dial_minutes_per_week=float(second.handle_min.sum() / W),
        second_dial_contact_rate=wilson(second.contact.sum(), len(second)),
        contact_after_overlap_rr=rr_after,
    )


def naive_history_rule(ov: dict) -> str:
    """What a dashboard reading would do: leads answer less after an overlap, so stop overlaps."""
    return "roll out with review" if ov["contact_after_overlap_rr"]["hi"] < 1 else "keep current"


# ----------------------------------------------------------------------------- the decision

def _switchback(L, outcome):
    wk = L.groupby(["week", "single_owner"])[outcome].mean().reset_index()
    t1 = wk[wk.single_owner][outcome].to_numpy(float)
    t0 = wk[~wk.single_owner][outcome].to_numpy(float)
    if len(t1) < 2 or len(t0) < 2 or t0.mean() == 0:
        return dict(rr=NAN, lo=NAN, hi=NAN, periods=[len(t1), len(t0)])
    m1, m0 = t1.mean(), t0.mean()
    v1, v0 = t1.var(ddof=1) / len(t1), t0.var(ddof=1) / len(t0)
    se = math.sqrt(v1 + v0)
    if se == 0:
        return dict(rr=m1 / m0, lo=NAN, hi=NAN, periods=[len(t1), len(t0)])
    df = (v1 + v0) ** 2 / (v1 ** 2 / (len(t1) - 1) + v0 ** 2 / (len(t0) - 1))
    q = sps.t.ppf(0.975, df)
    d = m1 - m0
    return dict(rr=m1 / m0, lo=(m0 + d - q * se) / m0, hi=(m0 + d + q * se) / m0, periods=[len(t1), len(t0)])


def decide_ownership(logs) -> dict:
    a = _prep(logs)
    design = logs["design"]["ownership"]
    kind = design.get("kind", "off")
    ov = overlaps(logs, a)
    base = dict(decision="Single next-contact owner per lead, across the three centers", design=design, history=ov)

    if kind in ("off", "all"):
        return dict(base, status="test only",
                    why="Every lead in these logs ran under one policy. The drop after an overlap fits two stories with "
                        "opposite decisions: the overlap annoys the lead, or cooling leads draw overlaps and the second "
                        "dial is a real extra chance. Nothing in the logs separates them.",
                    naive=naive_history_rule(ov), estimates={})

    L = lead_table(logs, a)
    if kind == "from_week":
        before = float(L[~L.single_owner].contacted.mean())
        after = float(L[L.single_owner].contacted.mean())
        return dict(base, status="comparison inadequate",
                    why="Ownership started on a date, so it is compared with earlier weeks. Anything else that moved "
                        "between those weeks (lead mix, reachability, staffing) is read as its effect.",
                    naive="roll out with review" if after > before else "keep current",
                    estimates=dict(before=before, after=after))

    if kind == "lead_random":
        strata = ["first_answered", "source"]
        est = {o: _rr(L, "single_owner", o, strata) for o in ["contacted", "quoted", "issued", "collected"]}
        est["quoted_per_contacted"] = _rr(L[L.contacted], "single_owner", "quoted", ["source"])
        interference = ("Both arms share the same agents: minutes one arm saves are spent partly on the other. "
                        "Confirm with a switchback before a full rollout.")
    elif kind == "switchback":
        est = {o: _switchback(L, o) for o in ["contacted", "quoted", "issued", "collected"]}
        est["quoted_per_contacted"] = _switchback(L[L.contacted], "quoted")
        interference = "Whole weeks switch together, so the arms do not compete for agents."
    else:
        raise ValueError(kind)

    minutes = {str(k): float(v) for k, v in L.groupby("single_owner").agent_min.mean().items()}
    c, q, qc = est["contacted"], est["quoted"], est["quoted_per_contacted"]
    if c["lo"] > 1 and q["rr"] > 1 and qc["lo"] > A.NONINFERIORITY:
        status = "roll out with review"
    elif c["hi"] < 1 or q["hi"] < 1:
        status = "keep current"
    else:
        status = "not yet"
    obs1 = L.loc[L.single_owner, "contacted"].mean()
    obs0 = L.loc[~L.single_owner, "contacted"].mean()
    return dict(base, status=status,
                why={"roll out with review": "Leads under a single owner are reached more often, and the extra contacts "
                                             "quote as often as the others.",
                     "keep current": "Leads under a single owner are reached less often: the other centers' dials were "
                                     "doing real work.",
                     "not yet": "The comparison is valid but the difference is still inside the noise."}[status],
                naive="roll out with review" if obs1 > obs0 else "keep current",
                estimates=est, agent_minutes_per_lead=minutes, interference=interference)


# ----------------------------------------------------------------------------- the control case

def decide_script(logs) -> dict:
    """A new closing script, randomized by lead. It acts at the close, so quotes and issued policies
    are counted before the stage it changes has played out: only collected payments can clear it."""
    if logs["design"]["script"].get("kind") != "lead_random":
        return dict(status="test only", estimates={}, naive="keep current")
    L = lead_table(logs)
    est = {o: _rr(L, "new_script", o, ["source"]) for o in ["quoted", "issued", "collected"]}
    q, col = est["quoted"], est["collected"]
    if col["lo"] > 1:
        status = "roll out with review"
    elif q["hi"] < 1 or col["hi"] < 1:
        status = "keep current"
    else:
        status = "not yet"
    naive = ("roll out with review" if L.loc[L.new_script, "quoted"].mean() > L.loc[~L.new_script, "quoted"].mean()
             else "keep current")
    return dict(status=status, estimates=est, naive=naive,
                why="The script acts at the close. A quote or an issued policy is counted before the stage it changes "
                    "has played out, so only first payments can clear it.")

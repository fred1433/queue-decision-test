"""Freeze what the page shows into JSON, and the published run's logs into CSV.

    python -m queuesim.build [bench.json]     # writes ../web/src/data/*.json and ../data/*

With a path, the bench results are read from it (a bench takes minutes); without, the bench runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from . import assumptions as A
from . import truth as T
from .decide import decide_ownership, lead_table
from .power import scale_ratio_table
from .simulate import Policy, World, simulate, weekly_totals

PUBLISHED_SEED = 2026
WEEKS = 8
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DATA = os.path.join(ROOT, "web", "src", "data")
DATA = os.path.join(ROOT, "data")
WORLDS = {"harm": World("harm", "harm"), "neutral": World("neutral", "neutral"), "rescue": World("rescue", "rescue")}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1, ensure_ascii=False)


def log_hash(logs) -> str:
    """SHA-256 over the tables the decision code reads (CSV bytes, so anyone can recompute it)."""
    m = hashlib.sha256()
    for k in ["leads", "attempts", "policies", "staffing", "spend"]:
        m.update(logs[k].to_csv(index=False).encode())
    return m.hexdigest()


def tapes(logs, n=5):
    """A few leads whose history shows an overlap: every dial, by center, with its outcome."""
    a = logs["attempts"]
    hum = a[a.handler != "AI"].sort_values(["lead_id", "t_min"])
    prev_c = hum.groupby("lead_id").handler.shift()
    prev_t = hum.groupby("lead_id").t_min.shift()
    ov = hum[(prev_c.notna()) & (prev_c != hum.handler) & ((hum.t_min - prev_t) <= A.COLLISION_WINDOW_MIN)]
    leads = logs["leads"].set_index("lead_id")
    cand = []
    for lid, g in a[a.lead_id.isin(ov.lead_id.unique())].groupby("lead_id"):
        arr = leads.loc[lid, "arrival_min"]
        day = (arr - 2 * 7 * 1440) / 1440
        if 5 <= len(g) <= 9 and 0 <= day < 3 and (g.t_min.max() - 2 * 7 * 1440) < 14 * 1440:
            cand.append(lid)
    rng = np.random.default_rng(7)
    pick = sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())
    out = []
    for lid in pick:
        g = a[a.lead_id == lid].sort_values("t_min")
        t0 = leads.loc[lid, "arrival_min"]
        out.append(dict(lead_id=int(lid), source=leads.loc[lid, "source"],
                        arrival_min=int(t0), week_start_min=2 * 7 * 1440,
                        dials=[dict(h=round((r.t_min - t0) / 60, 3), center=r.handler, outcome=r.outcome,
                                    overlap=bool(r.Index in ov.index)) for r in g.itertuples()]))
    return out


LEDGER = [
    # name, value, status, note
    ("Leads per week", "3,200", "assumed scale", "Used to size the synthetic operation, nothing else."),
    ("Quotes per week", "1,100", "assumed scale", "Calibration target for the synthetic funnel."),
    ("Issued policies per week", "40", "assumed scale", "Calibration target. No issue dates: no cohort can be built from it."),
    ("Collected policies per week", "3", "assumed scale", "Calibration target for a RATE; each synthetic week draws its own count."),
    ("Ad spend per week", "164,000 MXN", "assumed scale", "Split between Meta and Google by us."),
    ("Call center cost per week", "234,000 MXN", "assumed scale", "Split between three centers at hourly rates set by us."),
    ("Three centers dial the same leads; an AI voice agent takes overflow", "", "assumed scale", "Modelled as stated."),
    ("Center hourly rates", "165 / 135 / 110 MXN", "assumed", "Chosen so the three add up to the assumed total."),
    ("Contact, quote, issue and collection probabilities", "see truth.json", "assumed", "Tuned to land near the assumed counts."),
    ("Hour-of-day response curve", "see truth.json", "assumed", "People answer more after work."),
    ("Overlap rate between centers", f"{T.COLLISION_PROB:.0%} of dials", "assumed", "How often a second center dials within minutes."),
    ("What an overlap does to the lead", "three worlds", "not identified", "The subject of the test: history cannot tell."),
    ("Payment delays, cancellations, cohort maturity", "", "not identified", "Not modelled; the synthetic lag is fixed."),
    ("Fees, retained revenue, who pays which cost", "", "not identified", "Left out on purpose: no margin is computed."),
    ("Premium amounts", "lognormal by source", "assumed", "Only used to fill the policy table."),
]


def decision_record(own):
    """The Foundry-oriented measurement contract for this one decision (a proposed mapping, not a deployed action)."""
    return dict(
        object_type="AllocationDecision",
        decision_id="next-contact-owner-001",
        evidence_and_choice=dict(
            snapshot="synthetic, seed 2026, 8 weeks", cutoff="end of week 8", policy_version="multi-center retries v0",
            eligible_population="every new lead, all sources, all centers",
            baseline_metric="leads reached per lead", proposed_change="one next-contact owner per lead across centers",
            expected_effect="unknown sign: not identified from history", assumptions=["overlap mechanism unknown"],
        ),
        assignment_and_execution=dict(
            experimental_unit="lead (canonical lead ID across all centers)", arms=["current", "single owner"],
            assignment_probability=0.5, assignment_logged_at="lead creation, before any dial",
            eligible_but_untreated="logged with reason (no consent, outside contact window, duplicate)",
            approval="operations director", exposure="owner field honoured by every center's dialer",
            dialer_acknowledgement="per-dial check that the dialing center is the owner",
            stop_conditions=["leads reached per lead falls below current by more than 5% at week 2",
                             "any center dials outside the permitted window"],
        ),
        evaluation=dict(
            outcome_horizon="contact and quote: 14 days after lead creation; first payment: when mature",
            maturity_rule="a lead counts once 14 days have passed; a policy once its first due date has passed",
            cost_definition="agent minutes per lead, from dialer handle time",
            review_after_weeks=A.REVIEW_WEEKS, reversal_criterion="interval on leads reached per lead entirely below 1",
            observed_result=None, counterfactual_estimate=None,
        ),
    )


def main(bench_path: str | None = None):
    t0 = time.time()
    per_world = {}
    hashes = {}
    base_logs = None
    for name, w in WORLDS.items():
        hist, _ = simulate(WEEKS, PUBLISHED_SEED, Policy(), w)
        hashes[name] = log_hash(hist)
        if base_logs is None:
            base_logs = hist
        d_hist = decide_ownership(hist)
        exp, _ = simulate(WEEKS, PUBLISHED_SEED, Policy(ownership="lead_random"), w)
        d_exp = decide_ownership(exp)
        L = lead_table(exp)
        per_world[name] = dict(history_status=d_hist["status"], history_naive=d_hist["naive"],
                               experiment=dict(status=d_exp["status"], naive=d_exp["naive"], estimates=d_exp["estimates"],
                                               why=d_exp["why"], minutes=d_exp["agent_minutes_per_lead"],
                                               arms=dict(owner=dict(leads=int(L.single_owner.sum()),
                                                                    reached=int(L.loc[L.single_owner, "contacted"].sum())),
                                                         current=dict(leads=int((~L.single_owner).sum()),
                                                                      reached=int(L.loc[~L.single_owner, "contacted"].sum())))))
    hist_view = decide_ownership(base_logs)
    if bench_path:
        with open(bench_path) as f:
            bench = json.load(f)
    else:
        from .benchmark import run
        bench = run()
    demo = dict(
        meta=dict(seed=PUBLISHED_SEED, weeks=WEEKS, built_at=time.strftime("%Y-%m-%d"),
                  assumed_scale=T.CALIBRATION_WEEK, synthetic_week=weekly_totals(base_logs)),
        hashes=hashes, identical=len(set(hashes.values())) == 1,
        history=hist_view["history"], history_why=hist_view["why"], tapes=tapes(base_logs),
        worlds=per_world,
    )
    _dump(os.path.join(WEB_DATA, "demo.json"), demo)
    _dump(os.path.join(WEB_DATA, "bench.json"), bench)
    _dump(os.path.join(WEB_DATA, "planner.json"), dict(table=scale_ratio_table()))
    _dump(os.path.join(WEB_DATA, "ledger.json"), [dict(name=a, value=b, status=c, note=d) for a, b, c, d in LEDGER])
    _dump(os.path.join(WEB_DATA, "decision_record.json"), decision_record(hist_view))
    _dump(os.path.join(ROOT, "web", "public", "decision_record.json"), decision_record(hist_view))
    _dump(os.path.join(WEB_DATA, "truth.json"), {k: getattr(T, k) for k in dir(T) if k.isupper()})
    os.makedirs(DATA, exist_ok=True)
    for name in ["leads", "attempts", "policies", "staffing", "spend", "agents"]:
        base_logs[name].to_csv(os.path.join(DATA, f"history_{name}.csv.gz"), index=False, compression="gzip")
    print(f"built in {time.time() - t0:.0f}s; identical history across worlds: {demo['identical']}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

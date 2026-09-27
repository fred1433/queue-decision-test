"""The bench: the decision code, frozen, against every prespecified case, with frequencies.

Cases (each one a world the decision code never sees, and a design it does see in the logs):
  beneficial, no effect, harmful          three worlds with identical history under the current policy
  history only                            the same three worlds, no experiment: must not decide
  inadequate comparison                   ownership switched on at a date while leads drift
  switchback                              weeks alternate by lead arrival (retries still cross weeks)
  proxy reversal                          a closing script that lifts quotes and cuts collections
  proxy agreement                         a closing script that lifts both (the price of the rule above)
  harder mechanisms                       harm that wears off; overlaps that land on hard-to-reach leads;
                                          tighter capacity. Written with the bench, before its first run; no
                                          threshold of the decision code was set on them. They are not an
                                          independent hold-out: the code changed after runs that included them.

For each case: the true effect of a full rollout comes from oracle runs of the same world under both
policies (never from the parameters), and every replication is decided three ways: by the decision
code, by holding the current policy, and by the naive rule (highest observed rate).
Measures: rollout, keep, abstain; harmful recommendations; detection; interval coverage; decision loss.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .decide import decide_ownership, decide_script
from .simulate import Policy, World, simulate, violations, weekly_totals
from .stats import wilson

HARM = World("harm", "harm")
NEUTRAL = World("neutral", "neutral")
RESCUE = World("rescue", "rescue")

CASES = [
    # id, label, group, world, policy kwargs, decision ("ownership" | "script")
    ("beneficial", "Beneficial: overlaps annoy leads", "main", HARM, dict(ownership="lead_random"), "ownership"),
    ("no_effect", "No quote effect: the owner makes the dial", "main", NEUTRAL, dict(ownership="lead_random"), "ownership"),
    ("harmful", "Harmful: the second dial was a real chance", "main", RESCUE, dict(ownership="lead_random"), "ownership"),
    ("history_beneficial", "History only, beneficial world", "history", HARM, dict(ownership="off"), "ownership"),
    ("history_no_effect", "History only, no-effect world", "history", NEUTRAL, dict(ownership="off"), "ownership"),
    ("history_harmful", "History only, harmful world", "history", RESCUE, dict(ownership="off"), "ownership"),
    ("before_after", "Before/after while leads drift, no-effect world", "inadequate",
     World("neutral_drift", "neutral", reach_drift_per_week=-0.05), dict(ownership="from_week", ownership_from_week=4), "ownership"),
    ("switchback_beneficial", "Arrival-week cohorts, beneficial world", "switchback", HARM, dict(ownership="switchback"), "ownership"),
    ("switchback_harmful", "Arrival-week cohorts, harmful world", "switchback", RESCUE, dict(ownership="switchback"), "ownership"),
    ("proxy_reversal", "Closing script: more quotes, fewer collections", "proxy",
     World("script_reversal", script=(1.15, 1.10, 0.45)), dict(script_share=0.5), "script"),
    ("proxy_agreement", "Closing script: more quotes, more collections", "proxy",
     World("script_gain", script=(1.12, 1.0, 1.0)), dict(script_share=0.5), "script"),
    ("heldout_wears_off", "Annoyance that wears off after a day", "heldout",
     World("harm_wears_off", "harm", burn_hours=24), dict(ownership="lead_random"), "ownership"),
    ("heldout_on_reach", "Overlaps land on hard-to-reach leads (harmful)", "heldout",
     World("rescue_on_reach", "rescue", overlap_on_reach=True), dict(ownership="lead_random"), "ownership"),
    ("heldout_tight", "15% fewer agents, arms compete harder", "heldout",
     World("harm_tight", "harm", staffing_scale=0.85), dict(ownership="lead_random"), "ownership"),
]

WEEKS = 8
SHIP = "proceed to confirmation"
NAIVE_SHIP = "roll out"
ABSTAIN = {"not yet", "test only", "comparison inadequate"}


def _decide(logs, which):
    return decide_ownership(logs) if which == "ownership" else decide_script(logs)


def _rep(args):
    cid, world, pol, which, seed, weeks = args
    logs, _ = simulate(weeks, seed, Policy(**pol), world)
    d = _decide(logs, which)
    est = d.get("estimates", {})
    key = "contacted" if which == "ownership" else "collected"
    e = est.get(key, {}) if isinstance(est.get(key), dict) else {}
    return dict(case=cid, seed=seed, weeks=weeks, status=d["status"], naive=d["naive"],
                rr=e.get("rr"), lo=e.get("lo"), hi=e.get("hi"), guardrail=d.get("guardrail"), **violations(logs))


def _oracle(args):
    wname, world, which, arm, seed = args
    if which == "ownership":
        pol = Policy(ownership="all" if arm else "off")
    else:
        pol = Policy(script_share=1.0 if arm else 0.0)
    logs, _ = simulate(WEEKS, seed, pol, world)
    t = weekly_totals(logs)
    return dict(world=wname, which=which, arm=arm, seed=seed, **t)


def _mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return dict(mean=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se), n=int(len(x)))


def run(n_reps=40, n_oracle=30, workers=8, seed0=20_000, durations=(4, 8, 16), n_time=30):
    t0 = time.time()
    worlds = {}
    for c in CASES:
        worlds[(c[3].name, c[5])] = c[3]
    rep_jobs = [(c[0], c[3], c[4], c[5], seed0 + k, WEEKS) for c in CASES for k in range(n_reps)]
    time_jobs = [(f"time_{c[0]}_{w}", c[3], c[4], c[5], seed0 + 5000 + k, w)
                 for c in CASES if c[0] in ("beneficial", "harmful") for w in durations for k in range(n_time)]
    orc_jobs = [(wn, w, which, arm, seed0 + 90_000 + k) for (wn, which), w in worlds.items()
                for arm in (False, True) for k in range(n_oracle)]
    with ProcessPoolExecutor(workers) as ex:
        orc = list(ex.map(_oracle, orc_jobs, chunksize=2))
        reps = list(ex.map(_rep, rep_jobs, chunksize=2))
        times = list(ex.map(_rep, time_jobs, chunksize=2))

    truth = {}
    for (wn, which) in worlds:
        rows = [o for o in orc if o["world"] == wn and o["which"] == which]
        by = {(o["arm"], o["seed"]): o for o in rows}
        seeds = sorted({o["seed"] for o in rows})
        tr = {}
        for m in ["contacted_leads", "quotes", "issued", "collected", "agent_minutes"]:
            tr[m] = _mean_ci([by[(True, s)][m] - by[(False, s)][m] for s in seeds])
        tr["contacted_rr"] = float(np.mean([by[(True, s)]["contacted_leads"] / by[(True, s)]["leads"] for s in seeds])
                                   / np.mean([by[(False, s)]["contacted_leads"] / by[(False, s)]["leads"] for s in seeds]))
        tr["collected_rr"] = float(np.mean([by[(True, s)]["collected"] for s in seeds])
                                   / np.mean([by[(False, s)]["collected"] for s in seeds]))
        truth[f"{wn}|{which}"] = tr

    out_cases = []
    for c in CASES:
        cid, label, group, world, pol, which = c
        tr = truth[f"{world.name}|{which}"]
        value = tr["quotes"]["mean"] if which == "ownership" else tr["collected"]["mean"]
        unit = "quotes per week" if which == "ownership" else "collected policies per week"
        truth_rr = tr["contacted_rr"] if which == "ownership" else tr["collected_rr"]
        rs = [r for r in reps if r["case"] == cid]
        n = len(rs)

        def rate(pred):
            k = sum(1 for r in rs if pred(r))
            p, lo, hi = wilson(k, n)
            return dict(k=k, n=n, rate=p, lo=lo, hi=hi)

        metric = "quotes" if which == "ownership" else "collected"
        harmful_world = tr[metric]["hi"] < 0   # harmful only when the oracle interval sits below zero
        best = max(value, 0.0)

        def loss(chooser):
            return float(np.mean([best - (value if chooser(r) == SHIP else 0.0) for r in rs]))

        def finite(v):
            return v is not None and np.isfinite(float(v))
        covered = [r for r in rs if finite(r["lo"]) and finite(r["hi"])]
        cov = (sum(1 for r in covered if r["lo"] <= truth_rr <= r["hi"]) / len(covered)) if covered else None
        out_cases.append(dict(
            id=cid, label=label, group=group, decision=which, world=world.name, design=pol,
            true_effect=dict(value=value, unit=unit, detail=tr, rr=truth_rr),
            engine=dict(rollout=rate(lambda r: r["status"] == SHIP), keep=rate(lambda r: r["status"] == "keep current"),
                        abstain=rate(lambda r: r["status"] in ABSTAIN),
                        statuses={s: sum(1 for r in rs if r["status"] == s) for s in sorted({r["status"] for r in rs})}),
            naive=dict(rollout=rate(lambda r: r["naive"] == NAIVE_SHIP)),
            guardrails={g: sum(1 for r in rs if r.get("guardrail") == g) for g in ("collections", "collections unresolved", "sample ratio")},
            constraint_violations=dict(calls_outside_window=int(sum(r["calls_outside_window"] for r in rs)),
                                       leads_over_limit=int(sum(r["leads_over_limit"] for r in rs))),
            harmful_world=harmful_world,
            harmful_recommendation=dict(engine=rate(lambda r: r["status"] == SHIP and harmful_world),
                                        naive=rate(lambda r: r["naive"] == NAIVE_SHIP and harmful_world)),
            coverage=dict(rate=cov, n=len(covered), estimand="contacted leads, full rollout" if which == "ownership" else "collected, full rollout"),
            loss=dict(engine=loss(lambda r: r["status"]), hold=loss(lambda r: "keep current"),
                      naive=loss(lambda r: SHIP if r["naive"] == NAIVE_SHIP else "keep current"),
                      unit=unit),
        ))

    over_time = []
    for cid in ("beneficial", "harmful"):
        for w in durations:
            rs = [r for r in times if r["case"] == f"time_{cid}_{w}"]
            cnt = {s: sum(1 for r in rs if r["status"] == s) for s in [SHIP, "keep current", "not yet"]}
            over_time.append(dict(case=cid, weeks=w, n=len(rs), **cnt))

    return dict(n_reps=n_reps, n_oracle=n_oracle, weeks=WEEKS, seconds=round(time.time() - t0),
                truth=truth, cases=out_cases, over_time=over_time)


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    print(json.dumps(run(*args), indent=1, default=float))

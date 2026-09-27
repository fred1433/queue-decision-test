"""A synthetic insurance sales operation at an assumed scale.

`simulate(weeks, seed, policy, world)` returns:
  - `logs`: the tables an operation like this keeps (leads with their logged assignment, attempts,
    policies, staffing, spend) plus the design of any experiment that ran. This is all the decision
    code may read.
  - `hidden`: per-lead latent values no log records.

A `Policy` is what operations choose (who owns the next contact, whether a new closing script is
tested). A `World` is how the operation truly responds: it is the simulator's hidden truth.

The three main worlds differ ONLY in how leads would respond under single ownership. Under the
current multi-center policy their code paths are identical, so the same seed gives byte-identical
logs in all three (tests/test_worlds.py checks the hashes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import truth as T

WEEK_MIN = 7 * 1440
HANDLERS = ["A", "B", "C"]


@dataclass(frozen=True)
class World:
    """Hidden truth. `overlap` is what a second center's dial within minutes really does:
    - "harm":    it annoys the lead; later dials reach it less (coordination helps)
    - "neutral": the lead was cooling anyway; the owner would have made the same dial (no effect)
    - "rescue":  the lead was cooling anyway, and the other center's dial is a real extra chance
                 that an owner's normal cadence does not replace (coordination hurts)
    Held-out variants change mechanisms the engine was never tuned on.
    """
    name: str = "harm"
    overlap: str = "harm"
    burn_hours: float | None = None        # harm that wears off after this many hours
    overlap_on_reach: bool = False         # overlaps land on hard-to-reach leads
    staffing_scale: float = 1.0            # tighter capacity: arms compete for the same agents
    reach_drift_per_week: float = 0.0      # leads get harder to reach week after week
    script: tuple = (1.0, 1.0, 1.0)        # new closing script: quote, issue, collect multipliers


@dataclass
class Policy:
    staffing: dict = field(default_factory=lambda: dict(T.STAFFING))
    max_attempts: int = T.MAX_ATTEMPTS
    ai_enabled: bool = True
    # single next-contact owner per lead: "off", "all", "lead_random", "switchback", "from_week"
    ownership: str = "off"
    ownership_share: float = 0.5
    ownership_from_week: int = 4
    # a new closing script: share of leads that get it (0.5 = a test, 1.0 = everyone)
    script_share: float = 0.0
    leads_per_week: int = T.ASSUMED_SCALE["leads"]


def _agents(rng):
    rows = []
    for c, spec in T.CENTERS.items():
        for k in range(spec["agents"]):
            rows.append(dict(agent_id=f"{c}{k + 1:02d}", center=c,
                             tenure_months=float(np.round(rng.exponential(8.0) + 0.5, 1)),
                             script=str(rng.choice(["A", "B"]))))
    return pd.DataFrame(rows)


def _leads(rng, weeks, policy: Policy):
    rows = []
    arr_w = np.array(T.ARRIVAL_WEIGHTS, float)
    arr_w /= arr_w.sum()
    for w in range(weeks):
        for s, spec in T.SOURCES.items():
            base_n = policy.leads_per_week * spec["share"]
            n = rng.poisson(base_n)
            if s == "database":
                day = rng.integers(0, T.OPEN_DAYS, n)
                minute = 8 * 60 + rng.integers(0, 5, n)
            else:
                day = rng.integers(0, 7, n)
                hour = rng.choice(24, n, p=arr_w)
                minute = hour * 60 + rng.integers(0, 60, n)
            t = w * WEEK_MIN + day * 1440 + minute
            spend = spec["cpl"] * base_n
            for ti in t:
                rows.append((w, int(ti), s, spend / max(n, 1)))
    df = pd.DataFrame(rows, columns=["week", "arrival_min", "source", "cpl"])
    df = df.sort_values("arrival_min", kind="stable").reset_index(drop=True)
    df.insert(0, "lead_id", np.arange(len(df)))
    return df


def _assign(rng, leads, weeks, policy: Policy):
    """Logged assignment. Returns (owned: bool array, design dict)."""
    n = len(leads)
    wk = leads.week.to_numpy()
    kind = policy.ownership
    if kind == "off":
        return np.zeros(n, bool), dict(kind="off")
    if kind == "all":
        return np.ones(n, bool), dict(kind="all")
    if kind == "lead_random":
        owned = rng.random(n) < policy.ownership_share
        return owned, dict(kind="lead_random", unit="lead", share=policy.ownership_share)
    if kind == "switchback":
        order = rng.permutation(weeks)
        treated = set(int(x) for x in order[: weeks // 2])
        return np.isin(wk, list(treated)), dict(kind="switchback", unit="week", treated_weeks=sorted(treated))
    if kind == "from_week":
        return wk >= policy.ownership_from_week, dict(kind="from_week", unit="calendar", start_week=policy.ownership_from_week)
    raise ValueError(kind)


def simulate(weeks: int = 8, seed: int = 0, policy: Policy | None = None, world: World | None = None,
             tail_weeks: int = 2):
    """`weeks` of new leads, then `tail_weeks` more of dialing with no new leads, so every lead matures."""
    policy = policy or Policy()
    world = world or World()
    rng = np.random.default_rng(seed)
    agents = _agents(rng)
    roster = {c: agents[agents.center == c].to_dict("records") for c in HANDLERS}
    leads = _leads(rng, weeks, policy)
    n = len(leads)

    src = leads.source.to_numpy()
    arrival = leads.arrival_min.to_numpy()
    reach = rng.gamma(T.CONTACTABILITY_SHAPE, 1.0 / T.CONTACTABILITY_SHAPE, n)
    if world.reach_drift_per_week:
        reach = reach * (1 + world.reach_drift_per_week) ** leads.week.to_numpy()
    intent = rng.lognormal(0.0, T.INTENT_SIGMA, n)
    intent /= np.exp(T.INTENT_SIGMA ** 2 / 2)
    premium = np.array([T.SOURCES[s]["premium"] for s in src]) * rng.lognormal(0, T.PREMIUM_SIGMA, n)
    prio = np.array([T.PRIORITY[s] for s in src])
    src_contact = np.array([T.SOURCES[s]["contact"] for s in src])
    # a separate stream for experiment assignment, so that assignment never shifts the operation's draws
    arng = np.random.default_rng([seed, 1])
    owned, own_design = _assign(arng, leads, weeks, policy)
    script_arm = arng.random(n) < policy.script_share if policy.script_share > 0 else np.zeros(n, bool)
    if world.overlap_on_reach:
        overlap_p = np.clip(T.COLLISION_PROB / np.maximum(reach, 0.25), 0, 0.5)
        overlap_p *= T.COLLISION_PROB / overlap_p.mean()
    else:
        overlap_p = np.full(n, T.COLLISION_PROB)

    next_due = arrival.astype(float).copy()
    attempts = np.zeros(n, int)
    active = np.ones(n, bool)
    burn = np.ones(n)
    burn_until = np.full(n, np.inf)
    overlaps = np.zeros(n, int)
    att_rows = []
    quoted_by = {}
    staffing = {h: v * world.staffing_scale for h, v in policy.staffing.items()}

    def contact_p(i, t, hour, handler):
        age = max(t - arrival[i], 0.0)
        age_mult = T.AGE_FLOOR + (1 - T.AGE_FLOOR) * np.exp(-age / T.AGE_TAU_MIN)
        b = burn[i] if t < burn_until[i] else 1.0
        p = (T.BASE_CONTACT * src_contact[i] * T.HOUR_CONTACT[hour] * age_mult
             * T.ATTEMPT_FATIGUE ** attempts[i] * reach[i] * b)
        if handler == "AI":
            p *= T.AI["contact"]
        return min(p, 0.95)

    def quote_p(i, handler, agent):
        p = T.BASE_QUOTE * T.SOURCES[src[i]]["quote"] * intent[i] ** 0.3
        if handler == "AI":
            p *= T.AI["quote"]
        else:
            p *= T.CENTERS[handler]["quote"]
            if agent["tenure_months"] >= 6:
                p *= 1 + T.TENURE_QUOTE_GAIN
            if script_arm[i]:
                p *= world.script[0]
        return min(p, 0.97)

    def log(i, t, day, hour, handler, agent, outcome, hm, n_att):
        att_rows.append((i, int(t), day, hour, n_att, handler,
                         None if agent is None else agent["agent_id"],
                         None if agent is None else agent["tenure_months"],
                         None if agent is None else agent["script"], outcome, hm))

    def dial(i, t, day, hour, handler, agent):
        if rng.random() < contact_p(i, t, hour, handler):
            outcome = "quote" if rng.random() < quote_p(i, handler, agent) else "contact"
        else:
            outcome = "no_answer"
        hm = (T.AI_HANDLE_MIN if handler == "AI" else T.HANDLE_MIN)[outcome]
        log(i, t, day, hour, handler, agent, outcome, hm, int(attempts[i]) + 1)
        return hm, outcome

    def settle(i, t, outcome, handler):
        attempts[i] += 1
        if outcome == "quote":
            active[i] = False
            quoted_by[i] = (handler, t)
        elif outcome == "contact":
            active[i] = False
        elif attempts[i] >= policy.max_attempts:
            active[i] = False
        else:
            next_due[i] = t + T.RETRY_GAPS_MIN[min(attempts[i] - 1, len(T.RETRY_GAPS_MIN) - 1)]

    def cool(i, t):
        burn[i] *= T.COLLISION_BURN
        overlaps[i] += 1
        if world.burn_hours:
            burn_until[i] = t + world.burn_hours * 60

    for w in range(weeks + tail_weeks):
        for day in range(T.OPEN_DAYS):
            for hour in T.SLOTS:
                s = (w * 7 + day) * 1440 + hour * 60
                e = s + 60
                due = np.flatnonzero(active & (next_due < e))
                if due.size == 0:
                    continue
                new = due[attempts[due] == 0]
                old = due[attempts[due] > 0]
                new = new[np.lexsort((arrival[new], prio[new]))]
                old = old[np.argsort(next_due[old], kind="stable")]
                staff = staffing.get(hour, 0)
                cap = {c: staff * T.CENTER_SPLIT[c] * 60 * T.DIAL_SHARE * T.OCCUPANCY_CAP for c in HANDLERS}
                cap0 = sum(cap.values())
                used = 0.0
                overflow = []
                for i in list(new) + list(old):
                    left = [c for c in HANDLERS if cap[c] > 0]
                    if not left:
                        if attempts[i] == 0:
                            overflow.append(i)
                        continue
                    wts = np.array([cap[c] for c in left])
                    c = left[rng.choice(len(left), p=wts / wts.sum())]
                    agent = roster[c][rng.integers(len(roster[c]))]
                    t = min(max(next_due[i], s + (used / max(cap0, 1e-9)) * 60), e - 1)
                    hm, outcome = dial(i, t, day, hour, c, agent)
                    cap[c] -= hm
                    used += hm
                    # does a second center pick the same lead up within minutes?
                    overlap = rng.random() < overlap_p[i]
                    if overlap:
                        others = [o for o in HANDLERS if o != c and cap[o] > 0]
                        c2 = others[rng.integers(len(others))] if others else None
                        agent2 = roster[c2][rng.integers(len(roster[c2]))] if c2 else None
                        t2 = t + rng.uniform(1, 20)
                    if not overlap or c2 is None:
                        settle(i, t, outcome, c)
                        continue
                    if not owned[i]:
                        # current policy: the second center dials. Identical code in every world.
                        if outcome == "no_answer":
                            settle(i, t, outcome, c)
                            if active[i]:
                                hm2, out2 = dial(i, t2, day, hour, c2, agent2)
                                cap[c2] -= hm2
                                used += hm2
                                settle(i, t2, out2, c2)
                        else:
                            log(i, t2, day, hour, c2, agent2, "no_answer", T.HANDLE_MIN["no_answer"], int(attempts[i]) + 2)
                            cap[c2] -= T.HANDLE_MIN["no_answer"]
                            settle(i, t, outcome, c)
                            attempts[i] += 1
                        cool(i, t2)
                        continue
                    # single owner: the other center does not dial. What that means depends on the world.
                    settle(i, t, outcome, c)
                    if world.overlap == "harm":
                        continue                              # no second dial, no cooling
                    if world.overlap == "neutral":
                        if outcome == "no_answer" and active[i]:
                            hm2, out2 = dial(i, t2, day, hour, c, agent)   # the owner makes that dial
                            cap[c] -= hm2
                            used += hm2
                            settle(i, t2, out2, c)
                        cool(i, t2)
                        continue
                    if world.overlap == "rescue":
                        cool(i, t2)                           # the lead cools anyway; the extra chance is lost
                        continue
                    raise ValueError(world.overlap)
                if policy.ai_enabled:
                    for i in overflow:
                        t = min(max(next_due[i], s) + rng.uniform(0, 5), e - 1)
                        hm, outcome = dial(i, t, day, hour, "AI", None)
                        settle(i, t, outcome, "AI")

    attempts_df = pd.DataFrame(att_rows, columns=[
        "lead_id", "t_min", "day", "hour", "attempt_no", "handler", "agent_id",
        "agent_tenure_months", "script", "outcome", "handle_min"])
    attempts_df = attempts_df.sort_values(["t_min", "lead_id"], kind="stable").reset_index(drop=True)

    pol_rows = []
    for i, (h, t) in sorted(quoted_by.items()):
        s = src[i]
        mult = T.AI["issue"] if h == "AI" else T.CENTERS[h]["issue"]
        sm = world.script if (script_arm[i] and h != "AI") else (1.0, 1.0, 1.0)
        p_issue = min(T.BASE_ISSUE * T.SOURCES[s]["issue"] * mult * intent[i] * sm[1], 0.9)
        if rng.random() >= p_issue:
            continue
        p_cap = T.AI["capture"] if h == "AI" else T.CENTERS[h]["capture"]
        p_cap = min(p_cap * intent[i] ** T.CAPTURE_INTENT_POWER, 0.95)
        captured = rng.random() < p_cap
        base = T.COLLECT_IF_CAPTURED if captured else T.COLLECT_IF_LINK
        p_col = min(base * T.SOURCES[s]["collect"] * intent[i] ** T.COLLECT_INTENT_POWER * sm[2], 0.95)
        collected = rng.random() < p_col
        issued_t = t + rng.uniform(600, 4 * 1440)
        pol_rows.append((i, h, int(issued_t), round(float(premium[i]), 0), bool(captured), bool(collected)))
    policies = pd.DataFrame(pol_rows, columns=["lead_id", "handler", "issued_min", "premium_annual_mxn",
                                               "payment_on_call", "first_payment_collected"])
    policies = policies.sort_values(["issued_min", "lead_id"]).reset_index(drop=True)

    staff_rows = []
    for w in range(weeks):
        for day in range(T.OPEN_DAYS):
            for hour in T.SLOTS:
                for c in HANDLERS:
                    staff_rows.append((w, day, hour, c, staffing.get(hour, 0) * T.CENTER_SPLIT[c],
                                       T.CENTERS[c]["rate"]))
    staffing_df = pd.DataFrame(staff_rows, columns=["week", "day", "hour", "center", "agents", "rate_mxn_per_hour"])
    spend = (leads.groupby(["week", "source"]).agg(leads=("lead_id", "size"), spend_mxn=("cpl", "sum"))
             .reset_index())
    spend["spend_mxn"] = spend.spend_mxn.round(0)

    lead_log = leads[["lead_id", "week", "arrival_min", "source"]].copy()
    lead_log["single_owner"] = owned
    lead_log["new_script"] = script_arm
    design = dict(read_after_last_lead_days=7 * tail_weeks, ownership=own_design,
                  script=(dict(kind="lead_random", unit="lead", share=policy.script_share) if 0 < policy.script_share < 1
                          else dict(kind="all" if policy.script_share >= 1 else "off")))
    logs = dict(leads=lead_log, attempts=attempts_df, policies=policies, staffing=staffing_df, spend=spend,
                agents=agents, design=design, meta=dict(weeks=weeks, ai_cost_per_min=T.AI["cost_per_min"]))
    hidden = pd.DataFrame(dict(lead_id=np.arange(n), reach=reach, intent=intent, overlaps=overlaps,
                               burn=burn, premium=premium))
    return logs, hidden


def weekly_totals(logs) -> dict:
    """Per-week averages of the funnel and the costs, from logs only."""
    w = logs["meta"]["weeks"]
    a = logs["attempts"]
    p = logs["policies"]
    st = logs["staffing"]
    human_cost = float((st.agents * st.rate_mxn_per_hour).sum())
    ai_min = float(a.loc[a.handler == "AI", "handle_min"].sum())
    contacted = a.loc[a.outcome.isin(["contact", "quote"]), "lead_id"].nunique()
    return dict(
        leads=len(logs["leads"]) / w,
        contacted_leads=contacted / w,
        quotes=float((a.outcome == "quote").sum()) / w,
        issued=len(p) / w,
        collected=float(p.first_payment_collected.sum()) / w,
        ad_spend_mxn=float(logs["spend"].spend_mxn.sum()) / w,
        call_center_cost_mxn=(human_cost + ai_min * logs["meta"]["ai_cost_per_min"]) / w,
        agent_minutes=float(a.loc[a.handler != "AI", "handle_min"].sum()) / w,
    )

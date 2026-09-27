"""Freeze everything the page shows into JSON: the decision sheet for the published run, the
calendar of what the data can know, the grading against the hidden truth, and the truth itself.

    python -m queuesim.build            # writes ../web/src/data/*.json and ../data/*
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

from . import assumptions as A
from . import truth as T
from .decide import analyze, decision_objects
from .power import calendar
from .simulate import simulate, weekly_totals
from .verify import run as verify

PUBLISHED_SEED = 2026
WEEKS = 8
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DATA = os.path.join(ROOT, "web", "src", "data")
DATA = os.path.join(ROOT, "data")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1, ensure_ascii=False)


def truth_table():
    keep = {k: getattr(T, k) for k in dir(T) if k.isupper()}
    return keep


def main(n_reps=100, n_oracle=60):
    t0 = time.time()
    logs, hidden = simulate(WEEKS, PUBLISHED_SEED)
    result = analyze(logs)
    lv = {x["id"]: x for x in result["levers"]}
    cal = calendar(logs, collection_effect=lv["collection"]["expected"]["value"])
    decisions = decision_objects(result, week_label="2026-W39")
    grading = verify(n_reps=n_reps, n_oracle=n_oracle, hours_plan=lv["hours"]["plan"])
    meta = dict(
        published_seed=PUBLISHED_SEED, weeks=WEEKS, built_at=time.strftime("%Y-%m-%d %H:%M"),
        calibration_week=T.CALIBRATION_WEEK, simulated_week=weekly_totals(logs),
        assumptions={k: getattr(A, k) for k in dir(A) if k.isupper()},
        seconds=round(time.time() - t0, 1),
    )
    _dump(os.path.join(WEB_DATA, "sheet.json"), dict(meta=meta, economics=result["economics"], levers=result["levers"]))
    _dump(os.path.join(WEB_DATA, "calendar.json"), cal)
    _dump(os.path.join(WEB_DATA, "grading.json"), grading)
    _dump(os.path.join(WEB_DATA, "decisions.json"), decisions)
    _dump(os.path.join(WEB_DATA, "truth.json"), truth_table())
    # the published run's logs, so anyone can rerun the engine on exactly what it saw
    os.makedirs(DATA, exist_ok=True)
    for name in ["leads", "attempts", "policies", "staffing", "spend", "agents"]:
        logs[name].to_csv(os.path.join(DATA, f"{name}.csv.gz"), index=False, compression="gzip")
    hidden.to_csv(os.path.join(DATA, "hidden_truth_per_lead.csv.gz"), index=False, compression="gzip")
    print(f"built in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))

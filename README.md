# Same history, opposite effects: a test of one call-center allocation decision

A synthetic insurance sales operation at an assumed scale (three call centers dialing the same leads, an
AI voice agent on overflow), the decision code that reads its logs, and the bench that grades that code
against the simulator's hidden truth. Synthetic data only. Written by Frederic de Lavenne de Choulot; the
simulator and the decision code have the same author, so this is not an independent validation.

## The decision

Should each lead get one next-contact owner across the three centers, instead of any center calling it
back? Under the current process, three versions of the operation write byte-identical logs (by
construction: the versions differ only in code the current policy never runs; `tests/test_worlds.py`
checks the hashes), yet the change helps in one, barely moves quotes in one and hurts in one. On
history the decision code answers "test only" by rule (one policy in the logs, no comparison exists).
Once a randomized split runs, a first-stage screen separates them (proceed to confirmation, not yet, keep
current); the bench measures how often, and at what cost. It does not test a full rollout procedure.

## Layout

- `engine/queuesim/simulate.py`: the operation. `World` is the hidden truth, `Policy` what operations choose.
- `engine/queuesim/truth.py` (and `web/public/truth.json`): every hidden parameter.
- `engine/queuesim/decide.py`: the decision code. It reads logs only (`tests/test_isolation.py`).
- `engine/queuesim/benchmark.py`: every prespecified case, with frequencies of rollout, keep and abstain,
  rollouts where the change hurts, interval coverage and decision loss, against holding the current policy
  and against a naive rule.
- `engine/queuesim/power.py` and `web/src/lib/planner.ts`: a planner conditional on its assumptions.
- `web/src/data/*.json`: exactly what the page shows, rebuilt by `python -m queuesim.build bench/run4.json`.
- `data/history_*.csv.gz`: the published run's logs (seed 2026, 8 weeks, current policy).

## Bench runs, and everything that changed between them

- `bench/run1_before_maturity_rule.json`: first run. Leads from the last days were read before they could
  mature, so every switchback stayed undecided.
- Run 1 to run 2: a 14-day maturity rule in the decision code; two weeks of dialing after the last lead in
  the simulator; in the bench, "a rollout that hurts" now requires the true effect's whole interval below
  zero (it was: mean below zero). `bench/run2.json` carries a note: its harmful counts were recomputed
  from its stored results under that definition.
- Run 2 to run 3 (`bench/run3.json`, produced directly by the published `benchmark.py`), after an outside
  review: a guardrail on collected payments and a sample-ratio check in the decision code.
- Run 3 to run 4 (`bench/run4.json`), after a second review that executed the code: contacts and quotes
  counted within each lead's first 14 days on a logged extraction clock; exact collection inference that
  works with zero events (zero in both arms is reported as unresolved); the simulator's opening hours and
  10-call limit enforced and violations counted in the bench; a favorable screen now reads "proceed to
  confirmation" (the bench tests this first stage, not a full rollout procedure); a log-ratio interval
  for randomized arrival-week cohorts (approximate, with carryover). Enforcing the limits shifts later
  random draws, so the published seed gives a different run than before.
- The three harder mechanisms were written with the bench, before its first run. No threshold was set on
  them, but they are not an independent hold-out: the code changed after runs that included them.

## Run

```
python -m venv .venv && .venv/bin/pip install -r engine/requirements.txt
cd engine && ../.venv/bin/python -m pytest tests -q
../.venv/bin/python -m queuesim.benchmark 40 30 8 > ../bench/rerun.json   # 10 to 20 minutes on 8 cores
```

## History

Replayed from the working repository: only the folders above are kept (the page's own source is not);
the package's working name and a few words about where the scale came from were redacted; author
addresses were replaced by a no-reply address. Code, numbers, dates and order are otherwise original.

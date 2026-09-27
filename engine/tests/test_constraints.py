"""The simulator keeps its own operating limits: calls only inside opening hours, never more than the
attempt limit per lead, in every world and under every policy."""
from queuesim.simulate import Policy, World, simulate, violations

CASES = [(Policy(), World("harm", "harm")), (Policy(ownership="lead_random"), World("harm", "harm")),
         (Policy(ownership="lead_random"), World("neutral", "neutral")),
         (Policy(ownership="switchback"), World("rescue", "rescue")),
         (Policy(ownership="all"), World("harm_tight", "harm", staffing_scale=0.85))]


def test_no_call_outside_hours_and_no_lead_over_the_limit():
    for seed, (pol, world) in enumerate(CASES):
        v = violations(simulate(3, 40 + seed, pol, world)[0])
        assert v == dict(calls_outside_window=0, leads_over_limit=0), (pol.ownership, world.name, v)


def test_extraction_clock_is_logged():
    logs, _ = simulate(2, 5)
    assert logs["meta"]["extracted_at_min"] >= logs["attempts"].t_min.max()

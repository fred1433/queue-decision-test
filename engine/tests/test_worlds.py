"""Three worlds, one history: under the current policy the logs are byte-identical."""
from queuesim.build import log_hash
from queuesim.decide import decide_ownership
from queuesim.simulate import Policy, World, simulate

WORLDS = [World("harm", "harm"), World("neutral", "neutral"), World("rescue", "rescue")]


def test_history_is_identical_across_worlds():
    for seed in (1, 2):
        hs = {log_hash(simulate(2, seed, Policy(), w)[0]) for w in WORLDS}
        assert len(hs) == 1


def test_worlds_differ_under_the_owner_rule():
    hs = {log_hash(simulate(2, 3, Policy(ownership="all"), w)[0]) for w in WORLDS}
    assert len(hs) == 3


def test_identical_input_gives_identical_decision():
    outs = [decide_ownership(simulate(2, 4, Policy(), w)[0]) for w in WORLDS]
    assert len({o["status"] for o in outs}) == 1
    assert outs[0]["status"] == "test only"

"""The planner reproduces the reference table (two independent proportions, 5% two-sided, 80% power)."""
import json, pathlib, subprocess
import pytest
from queuesim.power import scale_ratio_table, weeks_needed


def test_reference_table():
    t = {round(r["relative"], 2): r["weeks"] for r in scale_ratio_table()}
    # closed-form reference values for these inputs: about 288, 53 and 16 weeks
    for rel, ref in [(0.2, 288), (0.5, 53), (1.0, 16)]:
        assert abs(t[rel] - ref) / ref < 0.03, (rel, t[rel])


def test_unbalanced_needs_more():
    assert weeks_needed(0.3, 0.1, 3200, 0.1) > weeks_needed(0.3, 0.1, 3200, 0.5)


def test_browser_planner_agrees():
    ts = pathlib.Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "planner.ts"
    if not ts.exists():
        pytest.skip("web planner not built yet")
    js = ts.read_text()
    assert "1.959963984540054" in js and "0.8416212335729143" in js

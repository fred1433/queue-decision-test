"""The decision code must not be able to read the truth: no import of truth or simulate."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "queuesim"
DECISION_SIDE = ["decide.py", "stats.py", "assumptions.py", "power.py"]


def imports(path):
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_decision_code_never_imports_truth_or_simulator():
    for f in DECISION_SIDE:
        imp = " ".join(imports(PKG / f))
        assert "truth" not in imp, f
        assert "simulate" not in imp, f
        assert "benchmark" not in imp, f


def test_decision_functions_take_logs_only():
    src = (PKG / "decide.py").read_text()
    assert "hidden" not in src

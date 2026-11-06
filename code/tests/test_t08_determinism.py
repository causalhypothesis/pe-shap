"""T8: the same seeds give the same result file, and a --fast run cannot overwrite a saved result."""
import json, pathlib, subprocess, sys
import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
DRIVER = CODE / "experiments" / "exp1_benchmarks.py"


def _payload(path):
    doc = json.loads(pathlib.Path(path).read_text())
    for k in ("date", "git", "versions", "runtime", "wall"):
        doc.get("meta", {}).pop(k, None)
    return doc


def test_same_seeds_same_json(tmp_path):
    if not DRIVER.exists():
        pytest.skip("exp1 driver absent")
    outs = []
    for i in range(2):
        out = tmp_path / f"exp1_{i}.json"
        subprocess.run([sys.executable, str(DRIVER), "--fast", "--out", str(out)], cwd=CODE, check=True)
        outs.append(_payload(out))
    assert outs[0]["meta"].get("mode") == "fast"
    assert outs[0] == outs[1]


def test_fast_cannot_overwrite_a_production_result(tmp_path):
    from pe_shap.io import PRODUCTION, write_result
    runs = [{"x": 1.0}]
    target = PRODUCTION / "t8_must_not_be_written.json"
    with pytest.raises(ValueError, match="must not write"):
        write_result(target, "s.py", {}, runs, mode="fast")
    assert not target.exists()
    write_result(tmp_path / "exp1.json", "s.py", {}, runs, mode="fast")
    assert (tmp_path / "exp1.json").exists()

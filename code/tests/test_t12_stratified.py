"""T12: the stratified variant of RowPinningSampler for the edge A -> B.  Every arm mean against exact
enumeration (the variant on {A} and {A, D}, plain row pinning on the other six coalitions); negative
controls at the cells whose exact gap exceeds 0.05.

The exact gaps (estimator limit minus the true arm mean, derived in conftest.py):
  plain row pinning (A overwritten, B as in the rows)    {A} and {A, D}: T=1,A=0 +0.156; T=0,A=0 +0.084;
                                                                      T=1,A=1 -0.140; T=0,A=1 -0.260
  the variant without the 1/P(A = a | T, W) weight       {A}: T=1,A=0 -0.077; T=0,A=0 -0.007 (not a control);
                                                                      T=1,A=1 +0.051
                                                         {A, D}: T=1,A=0 -0.090; T=0,A=0 -0.014 (not a control);
                                                                      T=1,A=1 +0.058
Also exp6_graphs.py's main() when a stratum is empty: one split undefined, and all undefined."""
import importlib.util, itertools, json, pathlib, sys
import numpy as np, pytest
from pe_shap.sampler import RowPinningSampler
from tests.conftest import N_BIG, t12_sample, t12_f, t12_mu, t12_plain, t12_unweighted_variant

MEDS = ["A", "B", "D"]
COALS = [S for k in range(4) for S in itertools.combinations(MEDS, k)]


@pytest.fixture(scope="module")
def X12():
    return t12_sample(N_BIG, seed=22)


def test_every_arm_mean(X12):
    B = RowPinningSampler(X12, "T", ["W"], MEDS, mediator_edges=[("A", "B")], strat_edge=("A", "B"))
    for ref in itertools.product((0.0, 1.0), repeat=3):
        for S in COALS:
            pins = {k: v for k, v in zip(MEDS, ref) if k in S}
            for tau in (1, 0):
                assert abs(B.mean_do(t12_f, tau, pins) - t12_mu(tau, pins)) <= 0.01, (ref, S, tau)
    assert B.min_stratum_ > 1000


def _variant_without_weight(sampler, X, tau, pins):
    """The arm's rows with A = a under the propensity weights only."""
    arm = X["T"].to_numpy(int) == tau
    keep = arm & (X["A"].to_numpy() == pins["A"])
    w = 1.0 / (sampler.p1_ if tau == 1 else 1 - sampler.p1_)
    rows = X.loc[keep].copy()
    for c, v in pins.items():
        rows[c] = v
    return float(np.sum(w[keep] * t12_f(rows)) / np.sum(w[keep]))


def test_negative_controls(X12):
    plain = RowPinningSampler(X12, "T", ["W"], MEDS)          # the edge not declared: plain row pinning
    n_controls = 0
    for S in (("A",), ("A", "D")):
        for tau, a, d in itertools.product((1, 0), (0.0, 1.0), (0.0, 1.0)):
            pins = {"A": a, **({"D": d} if "D" in S else {})}
            truth = t12_mu(tau, pins)
            for est, limit in ((plain.mean_do(t12_f, tau, pins), t12_plain(tau, pins)),
                               (_variant_without_weight(plain, X12, tau, pins), t12_unweighted_variant(tau, pins))):
                if abs(limit - truth) > 0.05:
                    n_controls += 1
                    assert abs(est - truth) > 0.03, (S, tau, pins, est, truth)
    assert n_controls >= 8


def _graph(priors, stratified=True):
    """One graph's record as exp6's one_graph() and run_split() leave it; priors None is an empty stratum."""
    v = None if priors is None else 0.01
    rec = {"ate": v, "lam_all": v, "phi": {"priors": priors, "juv": v, "charge": v}}
    if stratified:
        rec["min_stratum"] = 0 if priors is None else 50
    if priors is None:
        rec["undefined"] = "no rows"
    return rec


def _split(seed, g0, g2, g3):
    return {"seed": seed, "G0": _graph(g0, False), "G1": _graph(g0 - 0.01, False), "G2": _graph(g2), "G3": _graph(g3)}


@pytest.fixture
def run_exp6(monkeypatch, tmp_path, capsys):
    """exp6's main() on canned splits: data and model fits are replaced, summaries and printing are real."""
    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "exp6_graphs.py"
    spec = importlib.util.spec_from_file_location("exp6_graphs", path)
    exp6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp6)
    def run(*splits):
        out = tmp_path / "exp6.json"
        (tmp_path / "results").mkdir()                       # main() reads exp5's values; the test needs no saved result
        (tmp_path / "results" / "exp5.json").write_text(json.dumps({"runs": [{"seed": s, "phi": {}} for s in (0, 1)]}))
        monkeypatch.setattr(exp6, "CODE", tmp_path / "code")
        monkeypatch.setattr(exp6, "load_compas", lambda p: None)
        monkeypatch.setattr(exp6, "run_split", lambda df, seed, phi: splits[seed])
        monkeypatch.setattr(sys, "argv", ["exp6_graphs.py", "--fast", "--out", str(out)])
        exp6.main()
        return json.loads(out.read_text()), capsys.readouterr().out
    return run


def test_exp6_one_split_undefined(run_exp6):
    """G3 is compared with G0 on split 0 alone: 0.0037, not the 0.0463 of G3's mean against G0's over both."""
    doc, printed = run_exp6(_split(0, 0.10, 0.1025, 0.1037), _split(1, 0.20, 0.2025, None))
    d = doc["derived"]
    assert d["paired"]["G2"]["n_paired"] == 2 and d["paired"]["G2"]["change_priors"] == pytest.approx(0.0025)
    assert d["paired"]["G3"]["n_paired"] == 1 and d["paired"]["G3"]["change_priors"] == pytest.approx(0.0037)
    assert d["max_change_priors"] == pytest.approx(0.0037) and d["max_change_priors_up"] == 0.004
    assert doc["summary"]["G3.phi.priors"]["n"] == 1 and doc["summary"]["G3.phi.priors"]["n_missing"] == 1
    assert "G3 phi_priors 0.104" in printed and "G3 phi_priors none" in printed


def test_exp6_all_undefined(run_exp6):
    """No comparison is available: nulls, never 0."""
    doc, printed = run_exp6(_split(0, 0.10, None, None), _split(1, 0.20, None, None))
    d = doc["derived"]
    assert d["max_change_priors"] is None and d["max_change_priors_up"] is None
    for g in ("G2", "G3"):
        assert d["paired"][g] == {"n_paired": 0, "change_priors": None}
        assert doc["summary"][f"{g}.phi.priors"]["n"] == 0 and doc["summary"][f"{g}.phi.priors"]["mean"] is None
    assert printed.count("G2 phi_priors none") == 2 and printed.count("G3 phi_priors none") == 2

#!/usr/bin/env python3
"""EXP-6: sensitivity of the COMPAS PE-SHAP values to the assumed graph and to the age adjustment.

The 20 models of exp5_compas.py (same splits, seeds and model) and its reference defendant; four settings, each
estimated by row pinning on the test split:
  G0  the paper's graph: age pre-treatment, three parallel mediators; age weights (= exp5; checked equal).
  G1  adjustment sensitivity, not a graph: the age weights are dropped (age stays a model input), so the
      "effect of race" is the unadjusted arm contrast.
  G2  G0 plus the edge juv_fel_count -> priors_count; stratified variant for that edge.
  G3  G0 plus the edge priors_count -> felony; stratified variant for that edge.
The stratified variant is exact only if, given race and age, the added edge is the sole dependence between the
pinned parent and the free mediators (no unobserved common cause of that pair, which G0 allows).  The smallest
stratum is recorded; an empty stratum makes that split's values undefined (null, nullable leaves).
"derived": the change of mean phi_priors from G0 to G2 and to G3, each on the splits where that graph is defined,
with the paired count; the largest change, and that change rounded up to 3 decimals.  A comparison with no defined
split, and the maximum when no comparison is available, are null, never 0.
Usage (from code/): python experiments/exp6_graphs.py [--fast] [--out ../results/exp6.json]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
from pe_shap.data import load_compas, split                                          # noqa: E402
from pe_shap.sampler import RowPinningSampler, EmptyStratum                          # noqa: E402
from pe_shap import estimands as est                                                 # noqa: E402
from pe_shap.models import as_callable                                               # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402
from exp5_compas import fit_model, INPUTS, MEDS, REF, SHORT                          # noqa: E402

GRAPHS = {"G0": dict(pretreatment=["age"]),
          "G1": dict(pretreatment=[]),
          "G2": dict(pretreatment=["age"], mediator_edges=[("juv_fel_count", "priors_count")], strat_edge=("juv_fel_count", "priors_count")),
          "G3": dict(pretreatment=["age"], mediator_edges=[("priors_count", "felony")], strat_edge=("priors_count", "felony"))}
DEPS = ["experiments/exp6_graphs.py", "experiments/exp5_compas.py", "pe_shap/data.py", "pe_shap/sampler.py", "pe_shap/game.py",
        "pe_shap/estimands.py", "pe_shap/models.py", "pe_shap/io.py"]


def one_graph(te, f, spec):
    B = RowPinningSampler(te, "race", spec["pretreatment"], MEDS, mediator_edges=spec.get("mediator_edges", ()),
                          strat_edge=spec.get("strat_edge"))
    try:
        g = est.pe_game(B, f, {SHORT[m]: [m] for m in MEDS}, REF)
    except EmptyStratum as e:
        return {"ate": None, "lam_all": None, "phi": {k: None for k in SHORT.values()}, "min_stratum": B.min_stratum_, "undefined": str(e)}
    return {"ate": g["ate"], "lam_all": g["v"][tuple(sorted(SHORT.values()))], "phi": {k: g["phi"][k] for k in SHORT.values()},
            "min_stratum": B.min_stratum_}


def run_split(df, seed, exp5_phi):
    tr, te = split(df, seed, 0.3, "id")
    f = as_callable(fit_model(tr, seed), INPUTS, proba=True)
    te = te[INPUTS].reset_index(drop=True)
    out = {"seed": seed, **{name: one_graph(te, f, spec) for name, spec in GRAPHS.items()}}
    for k, v in exp5_phi.items():      # G0 is the game of exp5
        assert abs(out["G0"]["phi"][k] - v) < 1e-12, (seed, k)
    for name in ("G0", "G1"):
        out[name].pop("min_stratum")
    return out


def show(x):
    """A value on the progress line: "none" where the split's stratum was empty."""
    return "none" if x is None else f"{x:.3f}"


def compare_with_g0(runs):
    """The "derived" block; see the module docstring."""
    paired = {}
    for g in ("G2", "G3"):
        ok = [r for r in runs if r[g]["phi"]["priors"] is not None]
        mean = lambda name: sum(r[name]["phi"]["priors"] for r in ok) / len(ok)
        paired[g] = {"n_paired": len(ok), "change_priors": abs(mean(g) - mean("G0")) if ok else None}
    changes = [p["change_priors"] for p in paired.values() if p["change_priors"] is not None]
    ch = max(changes) if changes else None
    return {"max_change_priors": ch, "max_change_priors_up": None if ch is None else math.ceil(ch * 1000) / 1000,   # the text says "at most"
            "paired": paired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp6.json"))
    a = ap.parse_args()
    df = load_compas(CODE / "data" / "compas-scores-two-years.csv")
    phis = {r["seed"]: r["phi"] for r in json.loads((CODE.parent / "results" / "exp5.json").read_text())["runs"]}
    seeds = [0, 1] if a.fast else list(range(20))
    runs = []
    for s in seeds:
        t0 = time.time()
        rec = run_split(df, s, phis[s])
        runs.append(rec)
        print(f"split {s}: " + " | ".join(f"{g} phi_priors {show(rec[g]['phi']['priors'])}" for g in GRAPHS) +
              f" | strata {rec['G2']['min_stratum']}, {rec['G3']['min_stratum']} | {time.time() - t0:.0f}s", flush=True)
    nullable = tuple(f"{g}.{k}" for g in ("G2", "G3") for k in ("ate", "lam_all", "phi"))
    derived = compare_with_g0(runs)
    write_result(a.out, "experiments/exp6_graphs.py",
                 {"splits": seeds, "graphs": {k: {kk: (list(map(list, vv)) if kk == "mediator_edges" else vv) for kk, vv in v.items()}
                                              for k, v in GRAPHS.items()},
                  "exp5": "results/exp5.json", "reference": REF},
                 runs, extra={"derived": derived}, deps=DEPS, mode="fast" if a.fast else "full", nullable=nullable)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

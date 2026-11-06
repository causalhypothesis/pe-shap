#!/usr/bin/env python3
"""EXP-7: faithfulness on COMPAS, on the 20 models of exp5_compas.py (same splits, seeds and model).

Members: the reference defendant of exp5 (cohort medians: age 31, two priors, no juvenile felony, felony charge)
with race set to African-American ("aa") and to Caucasian ("c").  Score: the probability of the class predicted
at the member (pe_shap.faithfulness).  Masked mediators are replaced jointly by the values of each
held-out defendant of the member's group (all of them; age is not masked).
Rankings, each from its method's own definition, by absolute value:
  pe     PE-SHAP's phi (the exp5 game; checked equal to results/exp5.json)
  pw     PW-SHAP's Psi, matched donors, h = 0.4 (PWShapDonors, as exp5)
  last   the PE game's last positions v(all) - v(all \\ j)
  cs     Causal Shapley's phi of the mediators (causal_shapley_compas, all five inputs; per member;
         h = 0.4, with 0.2 and 0.8 as sensitivity)
Recorded per member: predicted probability and class; every masked subset's score; comprehensiveness and
sufficiency at k = 1, 2, 3 for each ranking and for the random baseline (mean over all 6 rankings); the
scores at k = 1 with each mediator ranked first, and the gain of priors-first over random (comprehensiveness: ranking
minus random; sufficiency: random minus ranking; positive = better than random); the donors' mean priors_count.
Usage (from code/): python experiments/exp7_faithfulness.py [--fast] [--out PATH]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
import numpy as np, pandas as pd                                                      # noqa: E402
from pe_shap.data import load_compas, split                                          # noqa: E402
from pe_shap.sampler import RowPinningSampler                                        # noqa: E402
from pe_shap import estimands as est                                                 # noqa: E402
from pe_shap.baselines.pwshap import PWShapDonors                                    # noqa: E402
from pe_shap.baselines.causal_shapley import causal_shapley_compas                   # noqa: E402
from pe_shap.faithfulness import faithfulness, rank_by_abs                           # noqa: E402
from pe_shap.models import as_callable                                               # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402
from exp5_compas import fit_model, INPUTS, MEDS, REF, SHORT                          # noqa: E402

MEMBERS = {"aa": 1.0, "c": 0.0}
HS = (0.2, 0.4, 0.8)
DEPS = ["experiments/exp7_faithfulness.py", "experiments/exp5_compas.py", "pe_shap/data.py", "pe_shap/sampler.py",
        "pe_shap/game.py", "pe_shap/estimands.py", "pe_shap/models.py", "pe_shap/io.py", "pe_shap/faithfulness.py",
        "pe_shap/baselines/pwshap.py", "pe_shap/baselines/causal_shapley.py"]


def model_and_test(df, seed):
    tr, te = split(df, seed, 0.3, "id")
    f = as_callable(fit_model(tr, seed), INPUTS, proba=True)
    return f, te[INPUTS].reset_index(drop=True)


def scores(subset, s0, order):
    """Comprehensiveness and sufficiency at k = 1..3 of a ranking, from the subset scores."""
    key = lambda B: "+".join(sorted(B)) or "none"
    return ({str(k): s0 - subset[key(order[:k])] for k in (1, 2, 3)},
            {str(k): s0 - subset[key(order[k:])] for k in (1, 2, 3)})


def final_run(df, seed, exp5_phi):
    f, te = model_and_test(df, seed)
    ref = dict(REF)
    B = RowPinningSampler(te, "race", ["age"], MEDS)
    players = {m: [m] for m in MEDS}
    g = est.pe_game(B, f, players, ref)
    allS = tuple(sorted(MEDS))
    for m in MEDS:      # the same game as exp5
        assert abs(g["phi"][m] - exp5_phi[SHORT[m]]) < 1e-12, (seed, m)
    last = {m: g["v"][allS] - g["v"][tuple(sorted(set(allS) - {m}))] for m in MEDS}
    D = PWShapDonors(te, "race", "age", MEDS, h=0.4, rule="matched", tolerance={"priors_count": 1})
    psi = {m: D.psi(f, ref, m) for m in MEDS}
    assert all(v is not None for v in psi.values()), seed
    ranks = {"pe": rank_by_abs(g["phi"]), "pw": rank_by_abs(psi), "last": rank_by_abs(last)}
    out = {"seed": seed, "ranking": {k: list(v) for k, v in ranks.items()},
           "priors_first": {k: float(v[0] == "priors_count") for k, v in ranks.items()}, "members": {}}
    for name, race in MEMBERS.items():
        x = {**ref, "race": race}
        donors = te[te["race"] == race]
        cs = {str(h): causal_shapley_compas(f, te, x, "race", "age", MEDS, h=h) for h in HS}
        r = faithfulness(f, x, MEDS, ranks["pe"], donors)
        subset = {(k or "none"): v for k, v in r["subset"].items()}
        s0 = subset["none"]
        own = {}
        for k, order in {**ranks, "cs": rank_by_abs({m: cs["0.4"]["phi"][m] for m in MEDS})}.items():
            c, s = scores(subset, s0, order)
            own[k] = {"comp": c, "suff": s}
        comp_r = {str(k): v for k, v in r["comp_random"].items()}
        suff_r = {str(k): v for k, v in r["suff_random"].items()}
        first = {}
        for m in MEDS:       # one mediator ranked first: comprehensiveness and sufficiency at k = 1
            rest = "+".join(sorted(set(MEDS) - {m}))
            first[SHORT[m]] = {"comp": s0 - subset[m], "suff": s0 - subset[rest]}
        out["members"][name] = {
            "p_hat": r["p_hat"], "cls": float(r["cls"]), "pred_no": float(r["cls"] == 0), "n_donors": len(donors),
            "donor_mean_priors": float(donors["priors_count"].mean()),
            "subset": subset, "comp_random": comp_r, "suff_random": suff_r, "own": own,
            "first": first,
            "gain": {"comp": first["priors"]["comp"] - comp_r["1"], "suff": suff_r["1"] - first["priors"]["suff"]},
            "cs": {h: {**{SHORT[m]: v["phi"][m] for m in MEDS}, "race": v["phi"]["race"], "age": v["phi"]["age"],
                       "priors_first": float(rank_by_abs({m: v["phi"][m] for m in MEDS})[0] == "priors_count"),
                       "efficiency_gap": sum(v["phi"].values()) - (v["f_x"] - v["v_empty"]), "min_rows": v["min_rows"]}
                   for h, v in cs.items()}}
        assert abs(out["members"][name]["cs"]["0.4"]["efficiency_gap"]) < 1e-10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or str(CODE.parent / "results" / "exp7.json")
    df = load_compas(CODE / "data" / "compas-scores-two-years.csv")
    e5 = json.loads((CODE.parent / "results" / "exp5.json").read_text())
    phis = {r["seed"]: r["phi"] for r in e5["runs"]}
    seeds = [0, 1] if a.fast else list(range(20))
    runs = []
    for s in seeds:
        t0 = time.time()
        rec = final_run(df, s, phis[s])
        runs.append(rec)
        m = rec["members"]
        print(f"split {s}: AA comp1 priors {m['aa']['first']['priors']['comp']:.3f} random {m['aa']['comp_random']['1']:.3f} | "
              f"C {m['c']['first']['priors']['comp']:.3f} {m['c']['comp_random']['1']:.3f} | rank pe {rec['ranking']['pe'][0]} "
              f"pw {rec['ranking']['pw'][0]} | {time.time() - t0:.0f}s", flush=True)
    config = {"splits": seeds, "members": {k: {**REF, "race": v} for k, v in MEMBERS.items()}, "exp5": "results/exp5.json",
              "design": "both members, all same-group test donors, rankings from each method's definition",
              "cs_h": list(HS)}
    write_result(out, "experiments/exp7_faithfulness.py", config, runs, deps=DEPS, mode="fast" if a.fast else "full")
    print("wrote", out)


if __name__ == "__main__":
    main()

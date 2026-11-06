#!/usr/bin/env python3
"""EXP-5: COMPAS on ProPublica's two-year file.

Graph (the authors' modelling assumption): race T and age are associated and neither causes the other;
priors_count, juv_fel_count and the felony indicator are mediators with parents race and age; no mediator
causes another; unobserved common causes of the mediators unrelated to race given age are allowed.
Protocol: load_compas (standard cohort filter, 5,278 defendants); 20 stratified 70/30 splits, seeds
0..19, person-disjoint asserted; an MLP classifier (100, 50) on standardised inputs as shipped (alpha = 0.001,
max_iter = 1000, random_state = the split's seed); inputs race, age and the three mediators.  Reference defendant:
the cohort medians (age 31, two priors, no juvenile felony, felony charge), fixed across splits.
  PE-SHAP  RowPinningSampler on the test split with pretreatment = age (logistic propensity on standardised
           age, clipped to [1e-3, 1 - 1e-3]): every coalition is valid because no mediator causes another, and
           both arms are standardised to the pooled age distribution.  The 8 coalitions, phi, lambda_all,
           CDE(all), the last position of priors_count, and the pair gap f(t, x) - f(t', x).
  PW-SHAP  PWShapDonors: donors of the same arm within h SDs of the reference age, matched on the other
           mediators (priors within +-1) / matched exactly / age only, h in {0.2, 0.4, 0.8}; stored as -Psi
           (sign reversed, as in the paper); undefined contrasts are null with the donor counts.
"derived": the share of the means phi_priors / lambda_all (%), the count of retrains with priors largest |phi|, and
the smallest and largest mean -Psi_priors over the donor rules, and the age windows in years (h times the test
split's age SD, PWShapDonors' unit, averaged over splits); the paper's text quotes these.
Variants: reference = per-split medians of the test split, and
"unadjusted" (no age weights).  run() takes any frame with the columns of load_compas.
Usage (from code/): python experiments/exp5_compas.py [--fast] [--out ../results/exp5.json]
"""
from __future__ import annotations
import argparse, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd                                                      # noqa: E402
from sklearn.neural_network import MLPClassifier                                     # noqa: E402
from sklearn.pipeline import make_pipeline                                           # noqa: E402
from sklearn.preprocessing import StandardScaler                                     # noqa: E402
from pe_shap.data import load_compas, split, mediator_dependence, COMPAS_MEDIATORS   # noqa: E402
from pe_shap.sampler import RowPinningSampler                                        # noqa: E402
from pe_shap import estimands as est                                                 # noqa: E402
from pe_shap.baselines.pwshap import PWShapDonors                                    # noqa: E402
from pe_shap.models import as_callable                                               # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402

MEDS = list(COMPAS_MEDIATORS)                                   # priors_count, juv_fel_count, felony
SHORT = {"priors_count": "priors", "juv_fel_count": "juv", "felony": "charge"}
INPUTS = ["race", "age"] + MEDS
REF = {"age": 31.0, "priors_count": 2.0, "juv_fel_count": 0.0, "felony": 1.0}
RULES = [("matched", 0.2), ("matched", 0.4), ("matched", 0.8), ("matched_exact", 0.4), ("age_only", 0.2), ("age_only", 0.4), ("age_only", 0.8)]
DEPS = ["experiments/exp5_compas.py", "pe_shap/data.py", "pe_shap/sampler.py", "pe_shap/game.py", "pe_shap/estimands.py",
        "pe_shap/models.py", "pe_shap/io.py", "pe_shap/baselines/pwshap.py"]


def fit_model(tr: pd.DataFrame, seed: int):
    """As shipped (compas_attribution.py): standardised inputs, MLP (100, 50), alpha 0.001, seeded by the split."""
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.001,
                                                      learning_rate="adaptive", max_iter=1000, random_state=seed))
    return m.fit(tr[INPUTS].to_numpy(float), tr["y"].to_numpy(int))


def run_split(df: pd.DataFrame, seed: int, reference: str = "fixed", adjust: bool = True, id_col: str | None = "id",
              pw_rules=RULES) -> dict:
    tr, te = split(df, seed, 0.3, id_col)
    model = fit_model(tr, seed)
    f = as_callable(model, INPUTS, proba=True)
    te = te[INPUTS + ["y"]].reset_index(drop=True)
    ref = dict(REF) if reference == "fixed" else {k: float(np.median(te[k])) for k in REF}
    B = RowPinningSampler(te[INPUTS], "race", ["age"] if adjust else [], MEDS)
    players = {SHORT[m]: [m] for m in MEDS}
    g = est.pe_game(B, f, players, ref)
    allS = tuple(sorted(players))
    lam_all = g["v"][allS]
    pair = f(pd.DataFrame([{**ref, "race": 1.0}]))[0] - f(pd.DataFrame([{**ref, "race": 0.0}]))[0]
    out = {"ate": g["ate"], "lam_all": lam_all, "cde_all": g["ate"] - lam_all,
           "phi": {k: g["phi"][k] for k in players},
           "last_priors": lam_all - g["v"][tuple(sorted(set(allS) - {"priors"}))],
           "pair_gap": float(pair), "pe_priors_largest": float(max(players, key=lambda k: abs(g["phi"][k])) == "priors"),
           "acc": float((model.predict(te[INPUTS].to_numpy(float)) == te["y"].to_numpy()).mean()),
           "majority": float(max(te["y"].mean(), 1 - te["y"].mean())), "clipped": float(B.clipped_share_), "n_test": len(te),
           "age_sd_test": float(te["age"].std(ddof=1)),
           "reference": ref}
    pw = {}
    for rule, h in pw_rules:
        D = PWShapDonors(te[INPUTS], "race", "age", MEDS, h=h, rule=rule, tolerance={"priors_count": 1})
        vals = {SHORT[m]: D.psi(f, ref, m) for m in MEDS}
        neg = {k: (None if v is None else -v) for k, v in vals.items()}
        key = f"{rule}_{h}"
        pw[key] = {**neg, "min_donors": {SHORT[m]: D.min_donors(m) for m in MEDS},
                   "priors_largest": (None if any(v is None for v in vals.values()) else
                                      float(max(vals, key=lambda k: abs(vals[k])) == "priors"))}
    out["pw"] = pw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp5.json"))
    a = ap.parse_args()
    df = load_compas(CODE / "data" / "compas-scores-two-years.csv")
    meds = {k: float(np.median(df[k])) for k in REF}
    assert meds == REF, (meds, REF)                    # the fixed reference is the cohort medians
    seeds = [0, 1] if a.fast else list(range(20))
    runs = []
    for s in seeds:
        t0 = time.time()
        rec = {"seed": s, **run_split(df, s)}
        rec.pop("reference")
        runs.append(rec)
        print(f"split {s}: ATE {rec['ate']:.3f} lam_all {rec['lam_all']:.3f} phi_priors {rec['phi']['priors']:.3f} "
              f"-Psi_priors(matched 0.4) {rec['pw']['matched_0.4']['priors']} | {time.time() - t0:.0f}s", flush=True)
    dep = {str(h): mediator_dependence(df, "race", "age", MEDS, REF["age"], h) for h in (0.2, 0.4, 0.8)}
    for rows in dep.values():
        for r in rows:
            r["pair"] = list(r["pair"])
    nullable = tuple(f"pw.{r}_{h}.{k}" for r, h in RULES for k in ("priors", "juv", "charge", "priors_largest"))
    mean = lambda xs: float(np.mean([x for x in xs if x is not None]))
    rule_means = [mean([r["pw"][f"{ru}_{h}"]["priors"] for r in runs]) for ru, h in RULES]
    derived = {"share_priors_pct": 100 * mean([r["phi"]["priors"] for r in runs]) / mean([r["lam_all"] for r in runs]),
               "pe_priors_largest_count": int(sum(r["pe_priors_largest"] for r in runs)),
               "pw_priors_rule_min": min(rule_means), "pw_priors_rule_max": max(rule_means),
               "window_years": {str(h): h * mean([r["age_sd_test"] for r in runs]) for h in (0.2, 0.4, 0.8)}}
    write_result(a.out, "experiments/exp5_compas.py",
                 {"splits": seeds, "test_size": 0.3, "reference": REF, "pw_rules": [f"{r}_{h}" for r, h in RULES],
                  "model": "StandardScaler + MLPClassifier(100, 50), alpha=0.001, max_iter=1000, random_state=seed",
                  "propensity": "logistic on standardised age, clip 1e-3", "n": len(df), "n_aa": int(df.race.sum()),
                  "n_c": int((1 - df.race).sum()), "label_rate": float(df.y.mean())},
                 runs, extra={"mediator_dependence": dep, "derived": derived}, deps=DEPS, mode="fast" if a.fast else "full", nullable=nullable)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

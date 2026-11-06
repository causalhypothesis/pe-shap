#!/usr/bin/env python3
"""EXP-8: German Credit with explicit encodings (female = 1, bad = 1; savings little 0, moderate 1,
quite rich 2, rich 3, missing 4; load_german).

Graph (as in the submission, after Bothmann et al.): sex T; age a pre-treatment input associated with sex; savings
and credit amount parallel mediators with parents sex and age.  Protocol as submitted: 20 stratified 50/50 splits,
seeds 0..19; the shipped MLP classifier (100, 50) on standardised inputs (alpha 0.001, adaptive, max_iter 1000,
random_state = the split's seed); inputs age, female, savings, amount in the shipped column order; output P(bad).  Reference pair: age 34,
savings "moderate" (1, the shipped code's value), amount 1,569, sex set to each arm.
  PE-SHAP  RowPinningSampler on the test split with age weights (primary) and without (sensitivity to the
           adjustment); the 4 coalitions; ATE, lambda_all, phi, which mediator has the larger |phi|; pair gap.
  PW-SHAP  PWShapDonors, sign reversed (-Psi): same-sex donors within h SDs of the reference age (age only,
           h = 0.4 and 0.8), and matched donors (amount within 0.4 SD of the test split for Psi_savings; savings
           exactly for Psi_amount; h = 0.4 and 0.8); undefined contrasts null with donor counts.
  Faithfulness for both members, all same-sex test donors: predicted P(bad) and class; the score with
           savings masked, amount masked, both masked; comprehensiveness at k = 1 for each mediator first.
  Accuracy (test, training, majority); mediator dependence within sex and age window (savings level without the
  missing code vs amount; missing-savings indicator vs amount).
Usage (from code/): python experiments/exp8_german.py [--fast] [--out ../results/exp8.json]
"""
from __future__ import annotations
import argparse, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd                                                      # noqa: E402
from scipy.stats import spearmanr                                                    # noqa: E402
from sklearn.neural_network import MLPClassifier                                     # noqa: E402
from sklearn.pipeline import make_pipeline                                           # noqa: E402
from sklearn.preprocessing import StandardScaler                                     # noqa: E402
from pe_shap.data import load_german, split                                          # noqa: E402
from pe_shap.sampler import RowPinningSampler                                        # noqa: E402
from pe_shap import estimands as est                                                 # noqa: E402
from pe_shap.baselines.pwshap import PWShapDonors                                    # noqa: E402
from pe_shap.faithfulness import faithfulness                                        # noqa: E402
from pe_shap.models import as_callable                                               # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402

MEDS = ["savings", "amount"]
INPUTS = ["age", "female"] + MEDS            # the shipped column order (it fixes the MLP's initial weights per input)
REF = {"age": 34.0, "savings": 1.0, "amount": 1569.0}
MEMBERS = {"woman": 1.0, "man": 0.0}
PW_RULES = [("age_only", 0.4), ("age_only", 0.8), ("matched", 0.4), ("matched", 0.8)]
DEPS = ["experiments/exp8_german.py", "pe_shap/data.py", "pe_shap/sampler.py", "pe_shap/game.py", "pe_shap/estimands.py",
        "pe_shap/models.py", "pe_shap/io.py", "pe_shap/faithfulness.py", "pe_shap/baselines/pwshap.py"]


def fit_model(tr: pd.DataFrame, seed: int):
    """As shipped (german_credit_attribution.py): standardised inputs, MLP (100, 50), alpha 0.001, seeded by the split."""
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.001,
                                                      learning_rate="adaptive", max_iter=1000, random_state=seed))
    return m.fit(tr[INPUTS].to_numpy(float), tr["y"].to_numpy(int))


def game(B, f):
    g = est.pe_game(B, f, {m: [m] for m in MEDS}, REF)
    lam = g["v"][tuple(sorted(MEDS))]
    return {"ate": g["ate"], "lam_all": lam, "phi": {m: g["phi"][m] for m in MEDS},
            "savings_larger": float(abs(g["phi"]["savings"]) > abs(g["phi"]["amount"]))}


def run_split(df, seed):
    tr, te = split(df, seed, 0.5)
    model = fit_model(tr, seed)
    f = as_callable(model, INPUTS, proba=True)
    te = te[INPUTS + ["y"]].reset_index(drop=True)
    X = te[INPUTS]
    pred = lambda D: model.predict(D[INPUTS].to_numpy(float))
    out = {"seed": seed,
           "weighted": game(RowPinningSampler(X, "female", ["age"], MEDS), f),
           "unweighted": game(RowPinningSampler(X, "female", [], MEDS), f),
           "pair_gap": float(f(pd.DataFrame([{**REF, "female": 1.0}]))[0] - f(pd.DataFrame([{**REF, "female": 0.0}]))[0]),
           "acc_test": float((pred(te) == te["y"].to_numpy()).mean()), "acc_train": float((pred(tr) == tr["y"].to_numpy()).mean()),
           "majority": float(max(te["y"].mean(), 1 - te["y"].mean()))}
    ref = {**REF, "female": 1.0}
    sd_amount = float(X["amount"].std(ddof=1))
    pw = {}
    for rule, h in PW_RULES:
        D = PWShapDonors(X, "female", "age", MEDS, h=h, rule=rule, tolerance={"amount": 0.4 * sd_amount})
        vals = {m: D.psi(f, ref, m) for m in MEDS}
        pw[f"{rule}_{h}"] = {**{m: (None if v is None else -v) for m, v in vals.items()},
                             "min_donors": {m: D.min_donors(m) for m in MEDS},
                             "savings_larger": (None if any(v is None for v in vals.values())
                                                else float(abs(vals["savings"]) > abs(vals["amount"])))}
    out["pw"] = pw
    out["members"] = {}
    for name, sex in MEMBERS.items():
        x = {**REF, "female": sex}
        r = faithfulness(f, x, MEDS, MEDS, te[te["female"] == sex][INPUTS])
        sub = {(k or "none"): v for k, v in r["subset"].items()}
        s0 = sub["none"]
        out["members"][name] = {"p_bad": r["p_hat"], "cls_bad": float(r["cls"]), "n_donors": int((te["female"] == sex).sum()),
                                "comp_savings": s0 - sub["savings"], "comp_amount": s0 - sub["amount"],
                                "comp_both": s0 - sub["amount+savings"],
                                "gap": (s0 - sub["savings"]) - (s0 - sub["amount"])}
    return out


def dependence(df, h=0.4):
    """Exploratory: Spearman rho within sex and the age window around 34 (window in SDs of the whole data's age)."""
    sd, rows = df["age"].std(ddof=1), []
    for sex in (0, 1):
        D = df[(df["female"] == sex) & ((df["age"] - REF["age"]).abs() <= h * sd)]
        lev = D[D["savings"] < 4]
        for name, a, b in (("savings level vs amount", lev["savings"], lev["amount"]),
                           ("savings missing vs amount", (D["savings"] == 4).astype(int), D["amount"])):
            rho, p = spearmanr(a, b)
            rows.append({"female": sex, "pair": name, "n": int(len(a)), "rho": float(rho), "p": float(p)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp8.json"))
    a = ap.parse_args()
    df = load_german(CODE / "data" / "german_credit.csv")
    seeds = [0, 1] if a.fast else list(range(20))
    runs = []
    for s in seeds:
        t0 = time.time()
        rec = run_split(df, s)
        runs.append(rec)
        print(f"split {s}: ATE {rec['weighted']['ate']:.3f} lam_all {rec['weighted']['lam_all']:.3f} phi_sav {rec['weighted']['phi']['savings']:.3f} "
              f"phi_amt {rec['weighted']['phi']['amount']:.3f} | acc {rec['acc_test']:.3f} | woman P(bad) {rec['members']['woman']['p_bad']:.3f} | "
              f"{time.time() - t0:.0f}s", flush=True)
    nullable = tuple(f"pw.{r}_{h}.{k}" for r, h in PW_RULES for k in ("savings", "amount", "savings_larger"))
    write_result(a.out, "experiments/exp8_german.py",
                 {"splits": seeds, "test_size": 0.5, "reference": REF, "members": MEMBERS, "pw_rules": [f"{r}_{h}" for r, h in PW_RULES],
                  "model": "StandardScaler + MLPClassifier(100, 50), alpha=0.001, max_iter=1000, random_state=seed",
                  "n": len(df), "n_women": int(df.female.sum()), "n_bad": int(df.y.sum()), "n_savings_missing": int((df.savings == 4).sum())},
                 runs, extra={"mediator_dependence": dependence(df)}, deps=DEPS, mode="fast" if a.fast else "full", nullable=nullable)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

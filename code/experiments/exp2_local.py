#!/usr/bin/env python3
"""EXP-2: local decompositions on S2 at x1 = (T, M1, M2) = (1, 0.50, 0.70) and x2 = (1, 1.50, 0.70).

Protocol as submitted (local_attributions_s2.py): R = 20 runs, seeds 1000 + r; N = 10,000 rows from S2
(scm.S2 reproduces the shipped generator exactly for the same seed); the shipped split with 200 rows held
out (98/2), 9,800 train; MLP regressor (64, 64), max_iter = 1000, random_state = 42.  At each sample, on
each run's model:
  PE-SHAP  MechanismSampler, CRN seed r: lambda_{M1}, lambda_{M2}, lambda_{12}, phi, with 20,000
           draws per coalition and the ATE with 200,000.
  oracle   the true mediator distributions with the same model, 20,000 draws.
  PW-SHAP  PWShapRegression, 20,000 draws: Psi_{M1}, Psi_{M2} in PW-SHAP's own sign.
  CS       CausalShapleyT, 20,000 draws: phi_T direct and indirect at the sample (t = 1).
--fast: R = 2, small budgets, meta.mode = "fast".  --ideal: the closed-form values at the two samples, next to
--out.  --print: the figure coordinates and the side-table cells from the result file.
Usage (from code/): python experiments/exp2_local.py [--fast] [--out ../results/exp2.json] [--print | --ideal]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np                                                                    # noqa: E402
from pe_shap import scm as scms, estimands as est, closed_forms as cf                 # noqa: E402
from pe_shap.sampler import MechanismSampler, make_mean_do                           # noqa: E402
from pe_shap.baselines.pwshap import PWShapRegression                                # noqa: E402
from pe_shap.baselines.causal_shapley import CausalShapleyT                          # noqa: E402
from pe_shap.models import fit_mlp_regressor, as_callable                            # noqa: E402
from pe_shap.io import write_result, pm, fmt                                         # noqa: E402

FULL = dict(R=20, J=20_000, J_ate=200_000, J_pw=20_000, J_cs=20_000)
FAST = dict(R=2, J=500, J_ate=2000, J_pw=500, J_cs=500)
N, N_EVAL, SEED0 = 10_000, 200, 1000
SAMPLES = {"x1": (0.5, 0.7), "x2": (1.5, 0.7)}
P = {"M1": ["M1"], "M2": ["M2"]}
COLS = ["T", "M1", "M2"]
DEPS = ["experiments/exp2_local.py", "pe_shap/scm.py", "pe_shap/sampler.py", "pe_shap/game.py", "pe_shap/estimands.py",
        "pe_shap/closed_forms.py", "pe_shap/models.py", "pe_shap/io.py", "pe_shap/baselines/pwshap.py",
        "pe_shap/baselines/causal_shapley.py"]


def shipped_split(D, seed):
    """_common.split_train_eval: a permutation with the run's seed; the first N_EVAL rows are held out."""
    D = D.iloc[np.random.default_rng(seed).permutation(len(D))].reset_index(drop=True)
    return D.iloc[N_EVAL:].reset_index(drop=True)


def one_run(r: int, b: dict) -> dict:
    scm = scms.S2()
    train = shipped_split(scm.sample(N, seed=SEED0 + r), SEED0 + r)
    f = as_callable(fit_mlp_regressor(train[COLS].to_numpy(float), train["Y"].to_numpy(float), seed=42), COLS)
    Xtr = train[COLS]
    A = MechanismSampler(Xtr, scm.graph, "T", [], scm.mediators, seed=r).fit()
    md = make_mean_do(A, f, b["J_ate"]); ate = md(1, {}) - md(0, {})
    O = scms.OracleSampler(scm, Xtr, J=b["J"], seed=r)
    ate_o = O.mean_do(f, 1, {}, J=b["J_ate"]) - O.mean_do(f, 0, {}, J=b["J_ate"])
    pw = PWShapRegression(Xtr, "T", scm.mediators, J=b["J_pw"], seed=r)
    cs = CausalShapleyT(A, "T", [], J=b["J_cs"], seed=r)
    out = {"ate": ate}
    for tag, (m1, m2) in SAMPLES.items():
        ref = {"M1": m1, "M2": m2}
        g, go = est.pe_game(A, f, P, ref, J=b["J"], ate=ate), est.pe_game(O, f, P, ref, J=b["J"], ate=ate_o)
        c = cs.phi_T(f, {"T": 1, **ref}, ["T", "M1", "M2"])
        out[tag] = {"lam": {"1": g["v"][("M1",)], "2": g["v"][("M2",)], "12": g["v"][("M1", "M2")]},
                    "phi": {"M1": g["phi"]["M1"], "M2": g["phi"]["M2"]},
                    "oracle": {"phi": {"M1": go["phi"]["M1"], "M2": go["phi"]["M2"]}, "lam12": go["v"][("M1", "M2")]},
                    "pw": {"psi1": pw.psi(f, ref, "M1"), "psi2": pw.psi(f, ref, "M2")},
                    "cs": {"dir": c["direct"], "ind": c["indirect"]}}
    return out


def ideal() -> dict:
    """Closed forms for the true outcome function at the two samples."""
    out = {}
    for tag, (m1, m2) in SAMPLES.items():
        g = cf.s2(m1, m2)
        out[tag] = {"lam1": g["lam"]["1"], "lam2": g["lam"]["2"], "lam12": g["lam"]["12"], "phi1": g["phi"][0], "phi2": g["phi"][1],
                    "psi1": g["psi"][0], "psi2": g["psi"][1], "cs_dir": g["cs_dir"][1], "cs_ind": g["cs_ind"][1]}
    return out


def print_values(path):
    S = json.loads(pathlib.Path(path).read_text())["summary"]
    E = ideal()
    print("% figure: mean, SD and printed label of each bar")
    for tag in SAMPLES:
        for k, lab in (("phi.M1", "phi_M1"), ("phi.M2", "phi_M2"), ("lam.12", "sum")):
            m, sd = S[f"{tag}.{k}"]["mean"], S[f"{tag}.{k}"]["sd"]
            print(f"{tag} {lab}: {m:.3f} sd {sd:.3f} label ${fmt(m, 2, sign=True)}$")
    print("% side table, camera-ready cells (ideal in brackets)")
    rows = (("lam.1", "lam1", 1), ("lam.2", "lam2", 1), ("lam.12", "lam12", 1), ("phi.M1", "phi1", 1), ("phi.M2", "phi2", 1),
            ("pw.psi1", "psi1", -1), ("pw.psi2", "psi2", -1), ("cs.dir", "cs_dir", 1), ("cs.ind", "cs_ind", 1))
    for k, e, sgn in rows:
        print(k, " | ".join(f"{tag}: ${pm(sgn * S[f'{tag}.{k}']['mean'], S[f'{tag}.{k}']['sd'], 3)}$ ({fmt(sgn * E[tag][e], 3, sign=True)})"
                            for tag in SAMPLES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp2.json"))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--ideal", action="store_true")
    a = ap.parse_args()
    if a.print:
        return print_values(a.out)
    if a.ideal:
        p = pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + "_ideal.json")
        p.write_text(json.dumps(ideal(), indent=1, sort_keys=True) + "\n"); return print("wrote", p)
    b = FAST if a.fast else FULL
    runs = []
    for r in range(b["R"]):
        t0 = time.time()
        rec = {"seed": SEED0 + r, **one_run(r, b)}
        runs.append(rec)
        print(f"run {r}: x1 phi ({rec['x1']['phi']['M1']:+.3f}, {rec['x1']['phi']['M2']:+.3f})  "
              f"x2 phi ({rec['x2']['phi']['M1']:+.3f}, {rec['x2']['phi']['M2']:+.3f}) | {time.time() - t0:.0f}s", flush=True)
    write_result(a.out, "experiments/exp2_local.py", {**b, "N": N, "n_eval": N_EVAL, "seeds": [SEED0 + r for r in range(b["R"])],
                 "samples": {k: [1.0, *v] for k, v in SAMPLES.items()}, "model": "MLPRegressor(64, 64), max_iter=1000, random_state=42"},
                 runs, deps=DEPS, mode="fast" if a.fast else "full")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

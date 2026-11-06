#!/usr/bin/env python3
"""Two interacting chains: the fitted rows of tab:two-chains, with the two paths as players.

TwoChains SCM: T ~ Bern(0.5); A1 -> A2 and B1 -> B2 binary chains; Y = 0.3 T + A2 B2 + N(0, 0.5^2).
R = 20 runs, seeds 0..19; N = 10,000 rows, 500 held out (a permutation with the run's seed), 9,500 train an
MLP regressor (64, 64), max_iter = 1000, random_state = 42, on (T, A1, A2, B1, B2).  Players A = {A1, A2} and
B = {B1, B2}.
  PE-SHAP  MechanismSampler along the chains, logistic mechanisms for the binary nodes, CRN seed r;
           2,000 draws per row and coalition, the ATE with 200,000.
  PW-SHAP  path-level effects Psi_A, Psi_B (their Suppl. G) with the block drawn jointly from the within-arm
           frequencies given the other path, without the graph (PWShapRegression), 20,000 draws;
           stored with the sign reversed (-Psi), as in the paper.
Reference sets: the untreated, the treated and all held-out rows.  The binary inputs take 16 configurations,
so each configuration is computed once per run and weighted by its count (the estimators are deterministic
given the reference).  --fast: R = 2 and small budgets.  --ideal: the exact rows (enumeration).
Usage (from code/): python experiments/exp_two_chains.py [--fast] [--out ../results/exp_two_chains.json] [--ideal]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np                                                                    # noqa: E402
from pe_shap import scm as scms, estimands as est, closed_forms as cf                 # noqa: E402
from pe_shap.sampler import MechanismSampler, make_mean_do                           # noqa: E402
from pe_shap.baselines.pwshap import PWShapRegression                                # noqa: E402
from pe_shap.models import fit_mlp_regressor, as_callable                            # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402

FULL = dict(R=20, J=2000, J_ate=200_000, J_pw=20_000)
FAST = dict(R=2, J=200, J_ate=2000, J_pw=500)
N, N_EVAL = 10_000, 500
MEDS = ["A1", "A2", "B1", "B2"]
COLS = ["T"] + MEDS
PLAYERS = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
DEPS = ["experiments/exp_two_chains.py", "pe_shap/scm.py", "pe_shap/sampler.py", "pe_shap/game.py", "pe_shap/estimands.py",
        "pe_shap/models.py", "pe_shap/io.py", "pe_shap/baselines/pwshap.py"]


def one_run(r: int, b: dict) -> dict:
    scm = scms.TwoChains()
    D = scm.sample(N, seed=r)
    D = D.iloc[np.random.default_rng(r).permutation(len(D))].reset_index(drop=True)
    held, train = D.iloc[:N_EVAL], D.iloc[N_EVAL:].reset_index(drop=True)
    f = as_callable(fit_mlp_regressor(train[COLS].to_numpy(float), train["Y"].to_numpy(float), seed=42), COLS)
    A = MechanismSampler(train[COLS], scm.graph, "T", [], MEDS, mechanism="logit", seed=r).fit()
    md = make_mean_do(A, f, b["J_ate"]); ate = md(1, {}) - md(0, {})
    pw = PWShapRegression(train[COLS], "T", MEDS, J=b["J_pw"], seed=r)
    per_config = {}
    for cfg in {tuple(int(v) for v in row) for row in held[MEDS].to_numpy()}:
        ref = dict(zip(MEDS, map(float, cfg)))
        g = est.pe_game(A, f, PLAYERS, ref, J=b["J"], ate=ate)
        per_config[cfg] = {"pe.A": g["phi"]["A"], "pe.B": g["phi"]["B"], "joint": g["v"][("A", "B")],
                           "pw.A": -pw.psi_block(f, ref, ["A1", "A2"]), "pw.B": -pw.psi_block(f, ref, ["B1", "B2"])}
    out = {"ate": ate}
    for name, rows in (("untreated", held[held["T"] == 0]), ("treated", held[held["T"] == 1]), ("all", held)):
        cfgs = [tuple(int(v) for v in row) for row in rows[MEDS].to_numpy()]
        avg = {k: float(np.mean([per_config[c][k] for c in cfgs])) for k in per_config[cfgs[0]]}
        out[name] = {"pw": {"A": avg["pw.A"], "B": avg["pw.B"], "sum": avg["pw.A"] + avg["pw.B"]},
                     "pe": {"A": avg["pe.A"], "B": avg["pe.B"]}, "joint": avg["joint"], "n_ref": len(cfgs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp_two_chains.json"))
    ap.add_argument("--ideal", action="store_true")
    a = ap.parse_args()
    if a.ideal:
        tc = cf.two_chains()
        doc = {g: {"pw": {"A": tc[k]["pw_A"], "B": tc[k]["pw_B"], "sum": tc[k]["pw_sum"]}, "pe": {"A": tc[k]["pe_A"], "B": tc[k]["pe_B"]},
                   "joint": tc[k]["joint"]} for g, k in (("untreated", "untreated"), ("treated", "treated"), ("both", "both"))}
        p = pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + "_ideal.json")
        p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n"); return print("wrote", p)
    b = FAST if a.fast else FULL
    runs = []
    for r in range(b["R"]):
        t0 = time.time()
        rec = {"seed": r, **one_run(r, b)}
        runs.append(rec)
        print(f"run {r}: untreated PW sum {rec['untreated']['pw']['sum']:.3f} joint {rec['untreated']['joint']:.3f} | "
              f"treated PW sum {rec['treated']['pw']['sum']:.3f} joint {rec['treated']['joint']:.3f} | {time.time() - t0:.0f}s", flush=True)
    write_result(a.out, "experiments/exp_two_chains.py", {**b, "N": N, "n_eval": N_EVAL, "seeds": list(range(b["R"])),
                 "model": "MLPRegressor(64, 64), max_iter=1000, random_state=42", "players": PLAYERS},
                 runs, deps=DEPS, mode="fast" if a.fast else "full")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

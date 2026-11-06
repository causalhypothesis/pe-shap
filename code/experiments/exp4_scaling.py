#!/usr/bin/env python3
"""EXP-4: larger mediator sets.

Design: T ~ Bern(0.5), M_k = alpha_k T + U_k, U_k ~ N(0, 0.5^2), Y = beta_0 T + sum_k beta_k M_k
+ sum_{k<l} gamma_kl M_k M_l + N(0, 0.5^2); zero reference; coefficients from configs/k10_coefficients.json and
k20_coefficients.json.  Data are drawn T first, then the U block row by row, then the outcome noise, so data
seed 11 reproduces the K = 10 run of test T10 and the K = 20 table.
  K = 10   data seeds 0..19, N = 20,000; row-pinning estimator (parallel mediators, no pre-treatment inputs);
           exact enumeration of the 2^10 coalitions (a) with the true function and (b) with an MLP (64, 64),
           random_state = 42, trained on the N rows; errors against the closed form, lambda_all, the RMS error
           of v(S) by the number of pinned mediators, wall times (enumeration; MLP fit).
  K = 20   the true function, N = 100,000, data seed 11; permutation sampling with P in {10, 30, 100, 300, 1000},
           five repeats with permutation seeds 100..104 and a fresh cache each; largest error (mean over repeats,
           SD with ddof = 1), distinct coalitions, wall time.  The coalition values use the same row pinning on
           numpy arrays (checked against RowPinningSampler on three coalitions before the run).
--fast: K = 10 with 2 seeds and K = 20 with P up to 30, mode "fast".
Usage (from code/): python experiments/exp4_scaling.py [--fast] [--out ../results/exp4.json]
"""
from __future__ import annotations
import argparse, itertools, os, pathlib, platform, subprocess, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd                                                      # noqa: E402
from pe_shap.scm import KDesign                                                       # noqa: E402
from pe_shap import closed_forms as cf                                                # noqa: E402
from pe_shap.sampler import RowPinningSampler, make_mean_do                          # noqa: E402
from pe_shap.game import coalition_table, shapley_exact, shapley_permutation          # noqa: E402
from pe_shap.models import fit_mlp_regressor, as_callable                            # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402

FULL = dict(seeds=list(range(20)), budgets=[10, 30, 100, 300, 1000], repeats=5)
FAST = dict(seeds=[0, 11], budgets=[10, 30], repeats=2)
N10, N20, SEED20 = 20_000, 100_000, 11
SIZES = (1, 3, 5, 7, 9, 10)
DEPS = ["experiments/exp4_scaling.py", "pe_shap/scm.py", "pe_shap/closed_forms.py", "pe_shap/sampler.py", "pe_shap/game.py",
        "pe_shap/models.py", "pe_shap/io.py", "configs/k10_coefficients.json", "configs/k20_coefficients.json"]


def simulate(K: KDesign, n: int, seed: int):
    """The data generator."""
    rng = np.random.default_rng(seed)
    T = rng.binomial(1, 0.5, n).astype(float)
    M = K.alpha[None, :] * T[:, None] + rng.normal(0.0, 0.5, (n, K.K))
    X = pd.DataFrame(M, columns=K.mediators); X.insert(0, "T", T)
    return X, K.f_true(X) + rng.normal(0.0, 0.5, n)


def hardware() -> str:
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor() or "unknown CPU"
    return f"{cpu}, {os.cpu_count()} cores, {platform.platform()}"


def k10_seed(K: KDesign, seed: int) -> dict:
    X, Y = simulate(K, N10, seed)
    kd = cf.kdesign(K.alpha, K.beta, K.gamma)
    idx = {m: k for k, m in enumerate(K.mediators)}
    players, ref = {m: [m] for m in K.mediators}, {m: 0.0 for m in K.mediators}
    B = RowPinningSampler(X, "T", [], K.mediators)
    out = {}
    fits = {"true": (K.f_true, 0.0)}
    t0 = time.time(); mlp = fit_mlp_regressor(X[["T"] + K.mediators].to_numpy(), Y, seed=42); fit_s = time.time() - t0
    fits["mlp"] = (as_callable(mlp, ["T"] + K.mediators), fit_s)
    for name, (f, fit_s) in fits.items():
        t0 = time.time()
        tab = coalition_table(make_mean_do(B, f), players, ref)
        phi = shapley_exact(tab["v"], tab["players"])
        enum_s = time.time() - t0
        err = np.array([abs(phi[m] - kd["phi"][idx[m]]) for m in K.mediators])
        rms = {}
        for s in SIZES:
            d = [tab["v"][S] - kd["v"]([idx[m] for m in S]) for S in tab["v"] if len(S) == s]
            rms[str(s)] = float(np.sqrt(np.mean(np.square(d))))
        out[name] = {"max_err": float(err.max()), "mean_err": float(err.mean()), "lam_all": float(tab["v"][tuple(sorted(K.mediators))]),
                     "rms_v": rms, "enum_s": enum_s, "fit_s": fit_s}
    return out


def k20(K: KDesign, budgets, repeats) -> dict:
    X, _ = simulate(K, N20, SEED20)
    truth = np.array(cf.kdesign(K.alpha, K.beta, K.gamma)["phi"])
    T, M = X["T"].to_numpy(), X[K.mediators].to_numpy()
    Mb, Mw = M[T == 1], M[T == 0]
    pairs = list(K.gamma.items())
    def f_np(t, Mm):
        y = K.beta0 * t + Mm @ K.beta
        for (k, l), g in pairs:
            y = y + g * Mm[:, k] * Mm[:, l]
        return y
    def gap(S):
        a, b = Mb.copy(), Mw.copy()
        for k in S:
            a[:, k] = 0.0; b[:, k] = 0.0
        return f_np(1.0, a).mean() - f_np(0.0, b).mean()
    ate = gap(())
    idx = {m: k for k, m in enumerate(K.mediators)}
    value = lambda S: ate - gap([idx[m] for m in S])
    # the numpy values equal RowPinningSampler's on a few coalitions
    rp = make_mean_do(RowPinningSampler(X, "T", [], K.mediators), K.f_true)
    for S in (("M1",), ("M2", "M7"), ("M3", "M5", "M11", "M20")):
        pins = {m: 0.0 for m in S}
        assert abs(value(S) - (ate - (rp(1, pins) - rp(0, pins)))) < 1e-10
    rows = []
    for P in budgets:
        errs, walls, dis = [], [], []
        for rep in range(repeats):
            t0 = time.time()
            phi, info = shapley_permutation(value, K.mediators, P, seed=100 + rep, cache={})
            walls.append(time.time() - t0)
            errs.append(float(np.max(np.abs(np.array([phi[m] for m in K.mediators]) - truth))))
            dis.append(info["n_distinct"])
        rows.append({"P": P, "err_mean": float(np.mean(errs)), "err_sd": float(np.std(errs, ddof=1)), "errs": errs,
                     "distinct": float(np.mean(dis)), "share_pct": float(np.mean(dis)) / 2 ** K.K * 100, "wall_mean": float(np.mean(walls))})
        print(f"K=20 P={P}: largest error {np.mean(errs):.4f} (sd {np.std(errs, ddof=1):.4f}), distinct {np.mean(dis):.0f}, {np.mean(walls):.1f}s", flush=True)
    return {"N": N20, "data_seed": SEED20, "lam_all": float(cf.kdesign(K.alpha, K.beta, K.gamma)["lam_all"]), "budgets": rows,
            "perm_seeds": [100 + r for r in range(repeats)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp4.json"))
    a = ap.parse_args()
    b = FAST if a.fast else FULL
    K10 = KDesign(CODE / "configs" / "k10_coefficients.json"); K20 = KDesign(CODE / "configs" / "k20_coefficients.json")
    runs = []
    for s in b["seeds"]:
        rec = {"seed": s, **k10_seed(K10, s)}
        runs.append(rec)
        print(f"K=10 seed {s}: true {rec['true']['max_err']:.4f} | MLP {rec['mlp']['max_err']:.4f} "
              f"(fit {rec['mlp']['fit_s']:.0f}s, enumeration {rec['mlp']['enum_s']:.0f}s)", flush=True)
    seed11 = next((r for r in runs if r["seed"] == 11), None)
    k20res = k20(K20, b["budgets"], b["repeats"])
    write_result(a.out, "experiments/exp4_scaling.py",
                 {**b, "N_k10": N10, "sizes": list(SIZES), "model": "MLPRegressor(64, 64), max_iter=1000, random_state=42",
                  "estimator": "row pinning, unweighted, zero reference"},
                 runs, extra={"k20": k20res, "hardware": hardware(),
                              "regression_seed11": {"true": seed11["true"]["max_err"], "mlp": seed11["mlp"]["max_err"]} if seed11 else None},
                 deps=DEPS, mode="fast" if a.fast else "full")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

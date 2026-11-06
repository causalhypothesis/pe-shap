#!/usr/bin/env python3
"""EXP-3: mediation structures, PW-SHAP and Causal Shapley as shipped, plus a PE-SHAP column.

The functions below, from model_outcomes_coal to run_one_iteration, are copied verbatim from the submission's
pw-causal-comparison/mediation_structures_comparison.py, with the same global seed
(np.random.seed(0)), case order, N = 500, 20 iterations and imputer settings, so the PW-SHAP and Causal Shapley
columns reproduce the shipped results file.  Data: T ~ Bern(0.5); C1 = q ~ U(0,1), or 3/5 U under T = 1 and
3/5 U + 2/5 under T = 0 when it mediates; C2 = d ~ Bern(0.5), or Bern(4/5 - 3/5 T) when it mediates; Y = q + d;
50/50 split; degree-2 polynomial regression.
Added (after the shipped computations of each iteration, using no random numbers, so the stream is unchanged):
PE-SHAP by row pinning on the test rows (no pre-treatment input, parallel mediators), the two Shapley values at
every test row as reference; the mean over references of the absolute value (as the shipped columns) and the
signed mean.  Exact values for the true function: phi_q = -0.4 and phi_d = -0.6 when mediating, 0 otherwise.
Common support: where C1 mediates it lies in [0, 0.6] under T = 1 and in [0.4, 1] under T = 0, so a reference value
of C1 occurs in both arms only in [0.4, 0.6].  Each run also records all three methods averaged over the test rows
in that interval, with the reference counts; the fits, the laws of the free inputs and the random stream are unchanged.
Usage (from code/): python experiments/exp3_structures.py [--fast] [--out ../results/exp3.json]
"""
from __future__ import annotations
import argparse, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np                                                                   # noqa: E402
import pandas as pd                                                                  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer                                          # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: F401,E402
from sklearn.preprocessing import PolynomialFeatures                                 # noqa: E402
from sklearn.pipeline import Pipeline                                                # noqa: E402
from pe_shap.sampler import RowPinningSampler                                        # noqa: E402
from pe_shap import estimands as est                                                 # noqa: E402
from pe_shap.io import write_result                                                  # noqa: E402

# ---------------------------------------------------------------- shipped (verbatim)


def model_outcomes_coal(model, coal, local_obs, ref_point, imp=None,
                        predict_proba=False):
    """Vectorized version. Returns a (n_obs, n) array of predictions."""
    n_obs, d = np.shape(local_obs)
    n = len(ref_point)

    coal = np.asarray(coal, dtype=float)
    local_obs = np.asarray(local_obs, dtype=float)

    # Build the input once: (n_obs, d) where present features = local_obs and
    # absent features = nan. The loop over reference points only changed via
    # imp.transform's posterior sampling, so we tile once and transform once.
    base = local_obs.copy()
    base[:, coal == 0] = np.nan

    # Tile to (n_obs * n, d) so the imputer (with sample_posterior=True)
    # produces n independent samples for each of the n_obs test points.
    tiled = np.tile(base, (n, 1))

    nan_cols = np.isnan(tiled).any(axis=0)
    if not nan_cols.any() or imp is None:
        # No imputation needed, or no imputer given — fall back to bootstrap
        # sampling from the reference set for the absent columns.
        if nan_cols.any():
            idx = np.random.randint(n, size=tiled.shape[0])
            sampled = ref_point[idx]
            tiled[:, nan_cols] = sampled[:, nan_cols]
        filled = tiled
    else:
        filled = imp.transform(tiled)

    if predict_proba:
        preds = model.predict_proba(filled)[:, 1]
    else:
        preds = model.predict(filled)

    return preds.reshape(n, n_obs).T  # -> (n_obs, n)


class IterativeImputerSubsets:
    def __init__(self, subsets=(), **kwargs):
        self.imps = [IterativeImputer(**kwargs) for _ in subsets]
        self.subsets = list(subsets)

    def fit(self, x_train):
        for imp, subset in zip(self.imps, self.subsets):
            imp.fit(x_train[:, subset])

    def transform(self, x):
        x_new = x
        for imp, subset in zip(self.imps, self.subsets):
            x_new[:, subset] = imp.transform(x[:, subset])
        return x_new


def decorrelate(x, mask):
    out = x.copy()
    idx = np.random.randint(len(x), size=len(x))
    sampled = x[idx]
    out[:, mask] = sampled[:, mask]
    return out


def generate_data(case, N):
    t = np.random.binomial(n=1, p=0.5 * np.ones(N))

    if case == 'none':
        q = np.random.uniform(size=N)
        d = np.random.binomial(n=1, p=0.5, size=N)
    elif case == 'q_only':
        q = np.random.uniform(size=N)
        q = t * (3 / 5) * q + (1 - t) * ((3 / 5) * q + 2 / 5)
        d = np.random.binomial(n=1, p=0.5, size=N)
    elif case == 'd_only':
        q = np.random.uniform(size=N)
        d = np.random.binomial(n=1, p=4 / 5 - 3 / 5 * t)
    elif case == 'both':
        q = np.random.uniform(size=N)
        q = t * (3 / 5) * q + (1 - t) * ((3 / 5) * q + 2 / 5)
        d = np.random.binomial(n=1, p=4 / 5 - 3 / 5 * t)
    else:
        raise ValueError(f"unknown case {case!r}")

    # Outcome matches the thesis description: Y = C_1 + C_2
    y = q + d
    return t, q, d, y


def run_one_iteration(case, N=200, imp_max_iter=10):
    t, q, d, y = generate_data(case, N)

    x = pd.DataFrame({'q': q, 'd': d, 't': t})
    N_train = N // 2
    x_train, y_train = x.iloc[:N_train], y[:N_train]
    x_test = x.iloc[N_train:]

    if y_train.std() == 0:
        return None

    # Imputers (lowered max_iter)
    imp = IterativeImputer(max_iter=imp_max_iter, random_state=0,
                           sample_posterior=True)
    imp.fit(x_train.values)

    outcome = Pipeline([('poly', PolynomialFeatures(2)),
                        ('lr', LinearRegression())])
    outcome.fit(x_train, y_train)

    xtr = x_train.values
    xte = x_test.values


    def cate_given_S(S_mask_qd):
        """S_mask_qd: length-2 bool array for (q, d).
        Returns per-sample CATE conditioning on those features."""
        coal = np.array([S_mask_qd[0], S_mask_qd[1], 1])  # T treated as in-coal,
                                                          # value forced via xte.

        xte_t1 = xte.copy(); xte_t1[:, 2] = 1
        preds_t1 = model_outcomes_coal(outcome, coal, xte_t1, xtr, imp=imp).mean(axis=1)

        xte_t0 = xte.copy(); xte_t0[:, 2] = 0
        preds_t0 = model_outcomes_coal(outcome, coal, xte_t0, xtr, imp=imp).mean(axis=1)

        return preds_t1 - preds_t0

    cate_full  = cate_given_S([1, 1])  # condition on (q, d)
    cate_q     = cate_given_S([1, 0])  # condition on q only
    cate_d     = cate_given_S([0, 1])  # condition on d only

    # Path-wise effects (per-sample):
    psi_q_per_sample = cate_full - cate_d   # adding q to {d}
    psi_d_per_sample = cate_full - cate_q   # adding d to {q}

    psi_q_ratio = np.abs(psi_q_per_sample).mean()
    psi_d_ratio = np.abs(psi_d_per_sample).mean()

    # Causal Shapley setup
    imp_no_t_full = IterativeImputerSubsets([[True, True, False]],
                                            max_iter=imp_max_iter,
                                            random_state=0,
                                            sample_posterior=True)
    imp_no_t_full.fit(xtr)

    x_do_q = decorrelate(xtr, [True, False, False])
    imp_do_q = IterativeImputer(max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_q.fit(x_do_q)
    imp_do_q_no_t = IterativeImputerSubsets([[True, True, False]], max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_q_no_t.fit(x_do_q)

    x_do_d = decorrelate(xtr, [False, True, False])
    imp_do_d = IterativeImputer(max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_d.fit(x_do_d)
    imp_do_d_no_t = IterativeImputerSubsets([[True, True, False]], max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_d_no_t.fit(x_do_d)

    # Per-sample Causal Shapley (return shape (n_obs,) for direct/indirect),
    # so we can take |.| per sample before averaging across the test set.
    def cs_terms(coal_with_t, coal_without_t, imp_a, imp_b):
        a = model_outcomes_coal(outcome, coal_with_t,    xte, xtr, imp=imp_a).mean(axis=1)
        b = model_outcomes_coal(outcome, coal_without_t, xte, xtr, imp=imp_a).mean(axis=1)
        c = model_outcomes_coal(outcome, coal_with_t,    xte, xtr, imp=imp_b).mean(axis=1)
        return c - b, a - c

    full_dir,  full_ind  = cs_terms([1, 1, 1], [1, 1, 0], None, None)
    qcoal_dir, qcoal_ind = cs_terms([1, 0, 1], [1, 0, 0], imp_do_q, imp_do_q_no_t)
    dcoal_dir, dcoal_ind = cs_terms([0, 1, 1], [0, 1, 0], imp_do_d, imp_do_d_no_t)
    empty_dir, empty_ind = cs_terms([0, 0, 1], [0, 0, 0], imp,      imp_no_t_full)

    phi_direct   = 1/3*full_dir   + 1/6*qcoal_dir   + 1/6*dcoal_dir   + 1/3*empty_dir
    phi_indirect = 1/3*full_ind   + 1/6*qcoal_ind   + 1/6*dcoal_ind   + 1/3*empty_ind

    run_one_iteration.last = (outcome, x_test)          # added: the fitted model and test rows for PE-SHAP
    run_one_iteration.per_sample = {"pw": {"q": np.abs(psi_q_per_sample), "d": np.abs(psi_d_per_sample)},
                                    "cs": {"direct": np.abs(phi_direct), "indirect": np.abs(phi_indirect)}}   # added: per test row
    return {
        'psi_q': psi_q_ratio,
        'psi_d': psi_d_ratio,
        'phi_direct':   np.abs(phi_direct).mean(),
        'phi_indirect': np.abs(phi_indirect).mean(),
    }


# ---------------------------------------------------------------- added: PE-SHAP on the same fitted model
CASES = [("none", "neither mediates", (0.0, 0.0)), ("q_only", "C1 mediates", (-0.4, 0.0)),
         ("d_only", "C2 mediates", (0.0, -0.6)), ("both", "both mediate", (-0.4, -0.6))]
C1_MEDIATES = ("q_only", "both")
COMMON_C1 = (0.4, 0.6)                                    # the values of C1 that occur in both arms when it mediates
DEPS = ["experiments/exp3_structures.py", "pe_shap/sampler.py", "pe_shap/game.py", "pe_shap/estimands.py", "pe_shap/io.py"]


def pe_shap_values(outcome, x_test):
    X = x_test.reset_index(drop=True)
    f = lambda D: outcome.predict(D[["q", "d", "t"]])
    B = RowPinningSampler(X, "t", [], ["q", "d"])
    ate = B.mean_do(f, 1, {}) - B.mean_do(f, 0, {})
    phi = {"q": [], "d": []}
    for _, row in X.iterrows():
        g = est.pe_game(B, f, {"q": ["q"], "d": ["d"]}, {"q": row["q"], "d": row["d"]}, ate=ate)
        for k in phi:
            phi[k].append(g["phi"][k])
    return ({k: {"abs_mean": float(np.mean(np.abs(v))), "mean": float(np.mean(v))} for k, v in phi.items()},
            {k: np.asarray(v) for k, v in phi.items()})


def common_support(x_test, per_sample, phi):
    """All three methods averaged over the reference rows whose C1 lies in COMMON_C1."""
    keep = x_test["q"].between(*COMMON_C1).to_numpy()
    return {"n_ref": int(keep.sum()), "n_ref_all": int(len(keep)),
            "pw": {k: float(v[keep].mean()) for k, v in per_sample["pw"].items()},
            "cs": {k: float(v[keep].mean()) for k, v in per_sample["cs"].items()},
            "pe": {k: {"abs_mean": float(np.abs(v[keep]).mean()), "mean": float(v[keep].mean())} for k, v in phi.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp3.json"))
    a = ap.parse_args()
    n_it = 2 if a.fast else 20
    np.random.seed(0)                                     # as shipped
    runs = []
    for case, label, exact in CASES:
        for it in range(n_it):
            t0 = time.time()
            res = run_one_iteration(case, N=500, imp_max_iter=10)
            if res is None:
                raise RuntimeError("degenerate draw")    # never happens with the shipped seed
            pe, phi = pe_shap_values(*run_one_iteration.last)
            runs.append({"case": case, "iteration": it, "pw": {"q": res["psi_q"], "d": res["psi_d"]},
                         "cs": {"direct": res["phi_direct"], "indirect": res["phi_indirect"]}, "pe": pe})
            if case in C1_MEDIATES:
                runs[-1]["common"] = common_support(run_one_iteration.last[1], run_one_iteration.per_sample, phi)
            print(f"{case} iter {it}: psi_q {res['psi_q']:.4f} psi_d {res['psi_d']:.4f} ind {res['phi_indirect']:.4f} | "
                  f"PE |q| {pe['q']['abs_mean']:.4f} |d| {pe['d']['abs_mean']:.4f} | {time.time() - t0:.1f}s", flush=True)
    summary_runs = []
    for it in range(n_it):
        rec = {"iteration": it}
        for case, _, _ in CASES:
            r = next(x for x in runs if x["case"] == case and x["iteration"] == it)
            rec[case] = {"pw": r["pw"], "cs": r["cs"], "pe": r["pe"], **({"common": r["common"]} if "common" in r else {})}
        summary_runs.append(rec)
    doc = write_result(a.out, "experiments/exp3_structures.py",
                 {"cases": {c: {"label": l, "exact_pe": list(e)} for c, l, e in CASES}, "N": 500, "split": "first half train, second half test",
                  "iterations": n_it, "seed": "np.random.seed(0) once, as shipped", "model": "PolynomialFeatures(2) + LinearRegression",
                  "common_support": {"C1": list(COMMON_C1), "settings": list(C1_MEDIATES)},
                  "shipped": "artifacts_full/NIPS-main/pw-causal-comparison/results/mediation_structures.txt"},
                 summary_runs, deps=DEPS, mode="fast" if a.fast else "full")
    S = doc["summary"]
    for case in C1_MEDIATES:
        for label, pre in (("all references", case), ("common support", f"{case}.common")):
            print(case, label, *(f"{k} {S[f'{pre}.{k}']['mean']:.3f}+-{S[f'{pre}.{k}']['sd']:.3f}"
                                 for k in ("pw.q", "pw.d", "pe.q.abs_mean", "pe.d.abs_mean", "cs.indirect")))
        print(case, "reference rows in the interval, per run: mean", S[f"{case}.common.n_ref"]["mean"], "of", S[f"{case}.common.n_ref_all"]["mean"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()

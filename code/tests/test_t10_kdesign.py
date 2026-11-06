"""T10: the K-mediator design.  shapley_exact on the closed-form game equals the formula
phi_k = a_k b_k + 1/2 sum_{l != k} g_kl a_k a_l; the single-draw K = 10 run (data seed 11) is reproduced."""
import json, pathlib
import numpy as np, pandas as pd, pytest
from pe_shap.scm import KDesign
from pe_shap.sampler import RowPinningSampler, make_mean_do
from pe_shap.game import coalition_table, shapley_exact, check_efficiency
from pe_shap.models import fit_mlp_regressor, as_callable
from pe_shap import closed_forms as cf
import itertools

CODE = pathlib.Path(__file__).resolve().parents[1]
K10 = KDesign(CODE / "configs" / "k10_coefficients.json")


def test_closed_form_game():
    kd = cf.kdesign(K10.alpha, K10.beta, K10.gamma)
    names = list(range(K10.K))
    v = {S: kd["v"](S) for k in range(K10.K + 1) for S in itertools.combinations(names, k)}
    phi = shapley_exact(v, names)
    assert max(abs(phi[k] - kd["phi"][k]) for k in names) <= 1e-12
    check_efficiency(phi, kd["lam_all"], tol=1e-12)


def _august_draw():
    """The K = 10 data with data seed 11, N = 20,000:
    T ~ Bern(0.5), U ~ N(0, 0.5^2) as one (n, K) block, Y = f + N(0, 0.5^2)."""
    rng = np.random.default_rng(11); n, K = 20_000, K10.K
    T = rng.binomial(1, 0.5, n).astype(float)
    M = K10.alpha[None, :] * T[:, None] + rng.normal(0.0, 0.5, (n, K))
    X = pd.DataFrame(M, columns=K10.mediators); X.insert(0, "T", T)
    return X, K10.f_true(X) + rng.normal(0.0, 0.5, n)


def _max_error(X, f):
    players, ref = {m: [m] for m in K10.mediators}, {m: 0.0 for m in K10.mediators}
    tab = coalition_table(make_mean_do(RowPinningSampler(X, "T", [], K10.mediators), f), players, ref)
    phi = shapley_exact(tab["v"], tab["players"])
    truth = cf.kdesign(K10.alpha, K10.beta, K10.gamma)["phi"]
    return max(abs(phi[m] - truth[k]) for k, m in enumerate(K10.mediators))


def test_august_run_true_function():
    X, _ = _august_draw()
    assert abs(_max_error(X, K10.f_true) - 0.006480) <= 1e-6        # that run: 0.006480318729393808


def test_august_run_mlp():
    X, Y = _august_draw()
    cols = ["T"] + K10.mediators
    f = as_callable(fit_mlp_regressor(X[cols].to_numpy(), Y, seed=42), cols)
    assert abs(_max_error(X, f) - 0.077874) <= 1e-3                 # that run: 0.07787402580628988

"""Shared fixtures: large samples of S1-S3 with the true function, probe points, tolerances,
and the exact arm means of the T11 and T12 designs (derived here, not typed)."""
import itertools
import numpy as np, pandas as pd, pytest
from pe_shap.scm import S1, S2, S3

PROBES = [(0.0, 0.0), (0.5, 0.7), (1.5, 0.7), (-0.5, 0.3), (1.0, -0.6)]
N_BIG = 400_000


@pytest.fixture(scope="session")
def big():
    out = {}
    for name, scm in (("S1", S1()), ("S2", S2()), ("S3", S3("submitted")), ("S3c", S3("clipped"))):
        X = scm.sample(N_BIG, seed=12345)
        out[name] = (scm, X, lambda D, s=scm: s.f_true(D))
    return out


# ---------------------------------------------------------------------------------- T11
# W ~ N(0,1); P(T=1|W) = expit(0.8 W); M_k = alpha_k T + delta_k W + U_k (two parallel mediators);
# f = T + 0.5 T W + M1 + M2 + 0.8 W M1 - 0.6 W M2.
T11_ALPHA, T11_DELTA = (0.5, 0.7), (0.6, -0.4)


def t11_sample(n, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, n)
    T = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.8 * W))).astype(float)
    M = {f"M{k+1}": T11_ALPHA[k] * T + T11_DELTA[k] * W + rng.normal(0, 0.5, n) for k in range(2)}
    return pd.DataFrame({"W": W, "T": T, **M})


def t11_f(D):
    W, T, M1, M2 = (D[c].to_numpy(float) for c in ("W", "T", "M1", "M2"))
    return T + 0.5 * T * W + M1 + M2 + 0.8 * W * M1 - 0.6 * W * M2


def t11_mu(tau, pins):
    """E[f | do(T=tau, M_S=m_S)] by integrating over W ~ N(0,1) and U: E[W] = 0, E[W^2] = 1, U independent
    of W with mean 0, so a free M_k has E[M_k] = alpha_k tau, E[W M_k] = delta_k; a pinned one m_k and 0."""
    EM, EWM = {}, {}
    for k, m in enumerate(("M1", "M2")):
        EM[m], EWM[m] = (pins[m], 0.0) if m in pins else (T11_ALPHA[k] * tau, T11_DELTA[k])
    return tau + 0.5 * tau * 0.0 + EM["M1"] + EM["M2"] + 0.8 * EWM["M1"] - 0.6 * EWM["M2"]


# ---------------------------------------------------------------------------------- T12
# W ~ Bern(0.5); P(T=1|W) = 0.3 + 0.4 W; A ~ Bern(0.2 + 0.3 T + 0.3 W); B ~ Bern(0.1 + 0.4 A + 0.2 T + 0.2 W);
# D ~ Bern(0.4 + 0.2 T - 0.2 W); f = 0.2 T + 0.5 A + 0.6 B + 0.3 D + 0.4 A B + 0.3 T W.  Edge A -> B.
def _pA(t, w): return 0.2 + 0.3 * t + 0.3 * w
def _pB(a, t, w): return 0.1 + 0.4 * a + 0.2 * t + 0.2 * w
def _pD(t, w): return 0.4 + 0.2 * t - 0.2 * w
def _bern(p, x): return p if x == 1 else 1 - p


def t12_sample(n, seed):
    rng = np.random.default_rng(seed)
    W = (rng.uniform(size=n) < 0.5).astype(float)
    T = (rng.uniform(size=n) < 0.3 + 0.4 * W).astype(float)
    A = (rng.uniform(size=n) < _pA(T, W)).astype(float)
    B = (rng.uniform(size=n) < _pB(A, T, W)).astype(float)
    D = (rng.uniform(size=n) < _pD(T, W)).astype(float)
    return pd.DataFrame({"W": W, "T": T, "A": A, "B": B, "D": D})


def t12_fv(t, a, b, d, w):
    return 0.2 * t + 0.5 * a + 0.6 * b + 0.3 * d + 0.4 * a * b + 0.3 * t * w


def t12_f(D):
    return t12_fv(*(D[c].to_numpy(float) for c in ("T", "A", "B", "D", "W")))


def t12_mu(tau, pins):
    """E[f | do(T=tau, pins)] by exact enumeration of the SCM (W keeps its law; B follows the A it gets)."""
    tot = 0.0
    for w, a, b, d in itertools.product((0, 1), repeat=4):
        p = 0.5
        p *= (a == pins["A"]) if "A" in pins else _bern(_pA(tau, w), a)
        p *= (b == pins["B"]) if "B" in pins else _bern(_pB(a, tau, w), b)
        p *= (d == pins["D"]) if "D" in pins else _bern(_pD(tau, w), d)
        tot += p * t12_fv(tau, a, b, d, w)
    return tot


def t12_plain(tau, pins):
    """Exact limit of PLAIN row pinning (propensity weights, pinned columns overwritten, every other column
    as in the arm's rows, so B keeps the law it has given the arm's own A)."""
    tot = 0.0
    for w, a0, b, d in itertools.product((0, 1), repeat=4):
        p = 0.5 * _bern(_pA(tau, w), a0) * _bern(_pB(a0, tau, w), b) * _bern(_pD(tau, w), d)
        tot += p * t12_fv(tau, pins.get("A", a0), pins.get("B", b), pins.get("D", d), w)
    return tot


def t12_unweighted_variant(tau, pins):
    """Exact limit of the stratified variant WITHOUT the 1/P(A = a | T, W) weight: the arm's rows with
    A = a under propensity weights only, so W keeps the tilt of the stratum."""
    a = pins["A"]
    num = den = 0.0
    for w, b, d in itertools.product((0, 1), repeat=3):
        p = 0.5 * _bern(_pA(tau, w), a) * _bern(_pB(a, tau, w), b) * _bern(_pD(tau, w), d)
        num += p * t12_fv(tau, a, b, pins.get("D", d), w)
        den += p
    return num / den

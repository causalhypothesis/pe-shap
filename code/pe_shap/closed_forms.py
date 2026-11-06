"""Exact values of the synthetic designs, as functions of the pins.

Derived from the structural equations of pe_shap.scm (independent zero-mean normal noise with variance
1/4, binary T with t = 1, t' = 0) with exact rationals.  Every function returns floats.

Keys of the per-design dicts:
  ate, nde, nie   ATE, natural direct effect, natural (total) indirect effect of the true function
  mu              arm means mu(tau, S) = E[f | do(T=tau, C_S=c_S)], keyed "0", "1", "2", "12"
                  (no pin, M1, M2, both) -> {"t": tau=1, "tprime": tau=0}
  cde, lam        CDE(c_S) and the portion eliminated lambda_S = ATE - CDE(c_S), same keys
  phi             PE-SHAP values (M1, M2) at the pins
  cond            PW-SHAP's conditional contrasts Delta_t(c_S) given the fixed set (pre-treatment
                  inputs held fixed), same keys
  psi, block, alt PW-SHAP's main effects (M1, M2), block effect of both mediators and the Suppl. F.2
                  alternatives (M1, M2), in PW-SHAP's own sign
  cs_dir, cs_ind  Causal Shapley's direct and indirect parts of T at the reference, {t: value}
"""
from __future__ import annotations

import itertools
from fractions import Fraction as Fr

from . import _two_chains_exact as tc

VAR = Fr(1, 4)                     # noise variance, 0.5 ** 2


def _q(x) -> Fr:
    """Exact rational from a pin given as int, Fraction or decimal float (0.7 -> 7/10)."""
    return Fr(str(x)) if isinstance(x, float) else Fr(x)


def _fl(x):
    if isinstance(x, dict):
        return {k: _fl(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_fl(v) for v in x)
    return float(x)


def _game(mu: dict, cond: dict) -> dict:
    """CDE, lambda, two-player Shapley values and PW-SHAP effects from the arm means and the
    conditional contrasts."""
    cde = {k: m["t"] - m["tprime"] for k, m in mu.items()}
    lam = {k: cde["0"] - c for k, c in cde.items() if k != "0"}
    phi = ((lam["1"] + lam["12"] - lam["2"]) / 2, (lam["2"] + lam["12"] - lam["1"]) / 2)
    psi = (cond["12"] - cond["2"], cond["12"] - cond["1"])
    return {"cde": cde, "lam": lam, "phi": phi, "psi": psi, "block": cond["12"] - cond["0"],
            "alt": (cond["1"] - cond["0"], cond["2"] - cond["0"])}


# ------------------------------------------------------------------------------------- S1
def s1(m1=0.0, m2=0.0) -> dict:
    """Chain T -> M1 -> M2, f = 0.5 T + M2; M1 = 0.8 T + U1, M2 = 0.7 M1 + U2."""
    m1, m2 = _q(m1), _q(m2)
    a1, a2, b1 = Fr(4, 5), Fr(7, 10), Fr(1, 2)
    mu = {"0": {"t": b1 + a1 * a2, "tprime": Fr(0)},
          "1": {"t": b1 + a2 * m1, "tprime": a2 * m1},
          "2": {"t": b1 + m2, "tprime": m2},
          "12": {"t": b1 + m2, "tprime": m2}}
    # PW-SHAP: f reads M2 only, and M2 depends on T only through M1, so every contrast with a
    # fixed mediator is b1; the empty one is the ATE.
    cond = {"0": b1 + a1 * a2, "1": b1, "2": b1, "12": b1}
    out = {"ate": b1 + a1 * a2, "nde": b1, "nie": a1 * a2, "mu": mu, "cond": cond, **_game(mu, cond),
           "cs_dir": {0: -b1 / 2, 1: b1 / 2},                       # b1 (t - 1/2)
           "cs_ind": {0: -a1 * a2 / 6, 1: a1 * a2 / 6}}             # (t - 1/2) a1 a2 / 3
    return _fl(out)


# ------------------------------------------------------------------------------------- S4
def s4(c1=0.7, c2=0.0) -> dict:
    """The S1 mechanisms with the linear model f = c0 T + c1 M1 + c2 M2 (S4: c1 = 0.7, c2 = 0).
    Appendix A.4: R^2 = b^2 s1^2 / (b^2 s1^2 + s2^2) = 49/149."""
    c1, c2 = _q(c1), _q(c2)
    a, b = Fr(4, 5), Fr(7, 10)
    R2 = b * b * VAR / (b * b * VAR + VAR)
    out = {"R2": R2,
           "phi": (c1 * a + c2 * a * b / 2, c2 * a * b / 2),
           "psi": (-(1 - R2) * c1 * a, Fr(0)),                       # PW-SHAP's own sign
           "alt": (-(c1 * a + c2 * a * b), -(c2 * a * b + R2 * c1 * a)),
           "block": -(c1 * a + c2 * a * b)}
    return _fl(out)


# ------------------------------------------------------------------------------------- S2
def s2(m1=0.0, m2=0.0) -> dict:
    """Parallel mediators M1 = 0.5 T + U1, M2 = 0.7 T + U2;
    f = 0.3 T + 0.2 M1 - 1.1 M2 + 0.2 M1 T + 0.8 M1 M2."""
    m1, m2 = _q(m1), _q(m2)
    a1, a2 = Fr(1, 2), Fr(7, 10)
    b1, b2, b3, b4, b5 = Fr(3, 10), Fr(1, 5), Fr(-11, 10), Fr(1, 5), Fr(4, 5)

    def f_mean(tau, e1, e2, e12):     # E[f | T = tau] given E[M1], E[M2], E[M1 M2]
        return b1 * tau + b2 * e1 + b3 * e2 + b4 * e1 * tau + b5 * e12

    mu = {}
    for tau, key in ((1, "t"), (0, "tprime")):
        E1, E2 = a1 * tau, a2 * tau
        mu.setdefault("0", {})[key] = f_mean(tau, E1, E2, E1 * E2)
        mu.setdefault("1", {})[key] = f_mean(tau, m1, E2, m1 * E2)
        mu.setdefault("2", {})[key] = f_mean(tau, E1, m2, E1 * m2)
        mu.setdefault("12", {})[key] = f_mean(tau, m1, m2, m1 * m2)
    # every coalition is closed (parallel mediators with no other cause than T): conditioning = fixing
    cond = {k: m["t"] - m["tprime"] for k, m in mu.items()}
    ate = mu["0"]["t"] - mu["0"]["tprime"]
    nde = b1 + b4 * 0                  # f(1, M(0)) - f(0, M(0)): E[M1(0)] = 0
    out = {"ate": ate, "nde": nde, "nie": ate - nde, "mu": mu, "cond": cond, **_game(mu, cond),
           # Causal Shapley (players T, M1, M2; T free ~ Bern(1/2))
           "cs_dir": {0: -Fr(7, 40) - m1 / 20, 1: Fr(3, 20) + m1 / 20},
           "cs_ind": {0: Fr(29, 240) - Fr(7, 150) * m1 - m2 / 30, 1: -Fr(23, 240) + Fr(7, 150) * m1 + m2 / 30}}
    return _fl(out)


# ------------------------------------------------------------------------------------- S3
def propensity(w, law="submitted"):
    lo, hi = (Fr(0), Fr(1)) if law == "submitted" else (Fr(1, 20), Fr(19, 20))
    return min(max(Fr(4, 5) - _q(w), lo), hi)


def s3(m1=0.0, m2=0.0, w=0.3, law="submitted") -> dict:
    """W ~ N(0.3, 1/4); T ~ Bern(p(W)); M1 = 0.5 T + U1; M2 = 0.7 M1 + 0.3 T + U2;
    f = W + T - 0.8 M1^2 - 0.5 T M1 + 0.4 M1 M2.  W enters additively, so it cancels from every
    contrast; the arm means use E[W] = 3/10.  The Causal Shapley parts depend on the law through
    p_w = P(T = 1 | W = w) at the reference."""
    m1, m2 = _q(m1), _q(m2)
    EW = Fr(3, 10)

    def f_mean(tau, e1, e11, e12):    # E[f | T = tau] given E[M1], E[M1^2], E[M1 M2]
        return EW + tau - Fr(4, 5) * e11 - Fr(1, 2) * tau * e1 + Fr(2, 5) * e12

    mu = {}
    for tau, key in ((1, "t"), (0, "tprime")):
        E1 = Fr(tau, 2); E11 = E1 ** 2 + VAR                         # M1 free
        mu.setdefault("0", {})[key] = f_mean(tau, E1, E11, Fr(7, 10) * E11 + Fr(3, 10) * tau * E1)
        mu.setdefault("1", {})[key] = f_mean(tau, m1, m1 ** 2, m1 * (Fr(7, 10) * m1 + Fr(3, 10) * tau))
        mu.setdefault("2", {})[key] = f_mean(tau, E1, E11, m2 * E1)   # pinning M2 leaves M1 free
        mu.setdefault("12", {})[key] = f_mean(tau, m1, m1 ** 2, m1 * m2)
    # PW-SHAP's contrasts: {M2} is not closed; given (T, M2) the arm's M1 is normal with
    # mean tau/2 + k (m2 - 13 tau/20) and variance v, k = 70/149, v = 25/149.
    k = Fr(7, 10) * VAR / (Fr(49, 100) * VAR + VAR)
    v = VAR - Fr(7, 10) * VAR * k
    cond2 = {}
    for tau in (1, 0):
        e1 = Fr(tau, 2) + k * (m2 - Fr(13, 20) * tau)
        cond2[tau] = f_mean(tau, e1, e1 ** 2 + v, m2 * e1) - EW
    cond = {"0": mu["0"]["t"] - mu["0"]["tprime"], "1": mu["1"]["t"] - mu["1"]["tprime"],
            "2": cond2[1] - cond2[0], "12": mu["12"]["t"] - mu["12"]["tprime"]}
    ate = cond["0"]
    nde = Fr(1)                        # f(1, M(0)) - f(0, M(0)) = 1 - 0.5 E[M1(0)] = 1
    p = propensity(w, law)
    out = {"ate": ate, "nde": nde, "nie": ate - nde, "mu": mu, "cond": cond, "p_w": p, **_game(mu, cond),
           # Causal Shapley, players W, T, M1, M2
           "cs_dir": {0: -Fr(5, 24) - Fr(11, 24) * p + m1 * (Fr(1, 24) + p / 6),
                      1: Fr(3, 4) - p / 2 + m1 * (p / 6 - Fr(5, 24))},
           "cs_ind": {0: Fr(41, 2400) + Fr(9, 400) * p - m1 * (p / 100 + Fr(1, 200)) - m2 * (p / 60 + Fr(1, 120)),
                      1: -Fr(59, 480) + Fr(77, 1200) * p + m1 * (Fr(3, 200) - p / 100) + m2 * (Fr(1, 40) - p / 60)}}
    return _fl(out)


def s3_e_p1mp(law="submitted") -> float:
    """E[p(W)(1 - p(W))] for W ~ N(0.3, 1/4) (numerical quadrature)."""
    import math
    from scipy.integrate import quad
    lo, hi = (0.0, 1.0) if law == "submitted" else (0.05, 0.95)
    dens = lambda w: math.exp(-((w - 0.3) ** 2) / (2 * 0.25)) / math.sqrt(2 * math.pi * 0.25)
    g = lambda w: (lambda p: p * (1 - p))(min(max(0.8 - w, lo), hi)) * dens(w)
    return quad(g, -4, 5, points=[-0.2, 0.8, -0.15, 0.75], limit=200)[0]


def s3_cs_means(law="submitted") -> dict:
    """Causal Shapley parts averaged over untreated and treated references (S3): with e = E[p(1-p)],
    E[p_w | T=0] = 2e, E[p_w | T=1] = 1 - 2e, and the mediators independent of W given T."""
    e = s3_e_p1mp(law)
    return {"e_p1mp": e,
            "untreated": {"dir": -5 / 24 - 11 / 12 * e, "ind": 41 / 2400 + 9 / 200 * e},
            "treated": {"dir": 11 / 48 + 5 / 6 * e, "ind": -61 / 1200 - 29 / 300 * e}}


# ------------------------------------------------------------------------------- two chains
def two_chains(variant=False) -> dict:
    """Paths A = {A1, A2} and B = {B1, B2} as players, by exact enumeration of the 16 binary
    configurations; f = 0.3 T + A2 B2 (variant: 0.3 T + (1 + T) A2 B2).  Averages over the
    untreated and the treated reference configurations and over both groups in equal proportion.
    PW-SHAP's path-level effects are returned sign-aligned (-Psi_B), as in tab:two-chains."""
    f = (lambda T, a1, a2, b1, b2: 0.3 * T + (1 + T) * a2 * b2) if variant else \
        (lambda T, a1, a2, b1, b2: 0.3 * T + a2 * b2)

    def cond_mean(tau, fixed):         # E[f | T = tau, fixed nodes]; A and B independent given T
        return tc.mean(f, tau, fixed)

    out = {}
    for tau, name in ((0, "untreated"), (1, "treated")):
        acc = {"pw_A": 0.0, "pw_B": 0.0, "pe_A": 0.0, "pe_B": 0.0, "joint": 0.0}
        for cfg, p in tc.law(tau, {}).items():
            a1, a2, b1, b2 = cfg
            pinsA, pinsB = {"A1": a1, "A2": a2}, {"B1": b1, "B2": b2}
            v, phi = tc.pe_values(f, {"A": pinsA, "B": pinsB})
            d_all = f(1, *cfg) - f(0, *cfg)
            d_noA = cond_mean(1, pinsB) - cond_mean(0, pinsB)
            d_noB = cond_mean(1, pinsA) - cond_mean(0, pinsA)
            acc["pw_A"] += p * -(d_all - d_noA); acc["pw_B"] += p * -(d_all - d_noB)
            acc["pe_A"] += p * phi["A"]; acc["pe_B"] += p * phi["B"]; acc["joint"] += p * v[("A", "B")]
        acc["pw_sum"] = acc["pw_A"] + acc["pw_B"]; acc["gap"] = acc["pw_sum"] - acc["joint"]
        out[name] = acc
    out["both"] = {k: (out["untreated"][k] + out["treated"][k]) / 2 for k in out["untreated"]}
    return out


# ---------------------------------------------------------------------------------- K-design
def kdesign(alpha, beta, gamma) -> dict:
    """Parallel mediators with pairwise interactions, zero reference (Appendix A.3):
    v(S) = sum_{k in S} a_k b_k + sum_{k<l} g_kl a_k a_l 1{S meets {k, l}},
    phi_k = a_k b_k + 1/2 sum_{l != k} g_kl a_k a_l.  gamma: {(k, l): g} with 0-based k < l."""
    K = len(alpha)
    phi = [alpha[k] * beta[k] for k in range(K)]
    for (k, l), g in gamma.items():
        phi[k] += 0.5 * g * alpha[k] * alpha[l]; phi[l] += 0.5 * g * alpha[k] * alpha[l]

    def v(S):
        S = set(S)
        return sum(alpha[k] * beta[k] for k in S) + sum(g * alpha[k] * alpha[l] for (k, l), g in gamma.items()
                                                         if k in S or l in S)
    return {"phi": phi, "lam_all": v(range(K)), "v": v}

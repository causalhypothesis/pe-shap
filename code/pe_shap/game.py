"""The portion-eliminated game and its Shapley values.

coalition_table  -- every coalition value of the game from one mean_do(tau, pins):
                    ATE, each CDE, each lambda_S = v(S), with the raw per-player and
                    joint values exposed separately from the allocation.
shapley_exact    -- exact Shapley value from a complete table (any K; 2^K values).
shapley_permutation -- permutation-sampling estimator with a coalition cache
                    (Castro, Gomez, Tejada 2009), used at K = 20.
check_efficiency -- |sum_j phi_j - v(all)| < 1e-10, asserted on every game.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


def _pins_for(coalition: tuple, players: dict, reference: dict) -> dict:
    pins = {}
    for p in coalition:
        for col in players[p]:
            pins[col] = reference[col]
    return pins


def coalition_table(mean_do, players: dict, reference: dict, t: int = 1, tprime: int = 0,
                    ate: float | None = None) -> dict:
    """All 2^K coalition values of v(S) = ATE - CDE_S(c_S).

    mean_do(tau, pins) -> E[f | do(T = tau, pins)].  `ate`: the ATE of the fitted model
    computed ONCE per run with the protocol's large budget (200,000 draws) and
    passed in; if None, the empty coalition is evaluated with mean_do's own budget (tests
    only).  Returns
      {"ate": float, "mu": {S: (mu_t, mu_tprime)}, "cde": {S: float}, "v": {S: float},
       "players": [...], "reference": {...}}
    with S a sorted tuple of player names; v[()] = 0 by definition.  Every value is
    checked finite.
    """
    names = sorted(players)
    mu, cde, v = {}, {}, {}
    for k in range(len(names) + 1):
        for S in itertools.combinations(names, k):
            if S == () and ate is not None:
                mu[S], cde[S] = (float("nan"), float("nan")), float(ate)
                continue
            pins = _pins_for(S, players, reference)
            a, b = mean_do(t, pins), mean_do(tprime, pins)
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(f"non-finite arm mean for coalition {S}: {a}, {b}")
            mu[S], cde[S] = (a, b), a - b
    ate = cde[()]
    for S in cde:
        v[S] = 0.0 if S == () else ate - cde[S]
    return {"ate": ate, "mu": mu, "cde": cde, "v": v, "players": names, "reference": dict(reference)}


def shapley_exact(v: dict, players: list) -> dict:
    """phi_j = sum_S |S|!(K-|S|-1)!/K! [v(S + j) - v(S)] over S not containing j."""
    K = len(players)
    phi = {}
    for j in players:
        others = [p for p in players if p != j]
        tot = 0.0
        for k in range(K):
            wgt = math.factorial(k) * math.factorial(K - k - 1) / math.factorial(K)
            for S in itertools.combinations(others, k):
                S = tuple(sorted(S))
                Sj = tuple(sorted(S + (j,)))
                tot += wgt * (v[Sj] - v[S])
        phi[j] = tot
    return phi


def shapley_permutation(value_fn, players: list, n_perm: int, seed: int = 0, cache: dict | None = None):
    """Permutation-sampling Shapley values.

    value_fn(S: sorted tuple) -> v(S); evaluated once per distinct coalition thanks to
    `cache` (pass a dict to keep it across budgets).  Returns (phi, info) with
    info = {"n_distinct": number of coalitions evaluated, "n_perm": n_perm}.
    """
    rng = np.random.default_rng(seed)
    cache = {} if cache is None else cache

    def val(S):
        S = tuple(sorted(S))
        if S not in cache:
            cache[S] = 0.0 if S == () else value_fn(S)
        return cache[S]

    acc = {p: 0.0 for p in players}
    for _ in range(n_perm):
        order = list(rng.permutation(players))
        prev, S = 0.0, []
        for p in order:
            S.append(p)
            cur = val(S)
            acc[p] += cur - prev
            prev = cur
    phi = {p: acc[p] / n_perm for p in players}
    return phi, {"n_distinct": len(cache), "n_perm": n_perm}


def check_efficiency(phi: dict, v_all: float, tol: float = 1e-10) -> float:
    vals = list(phi.values()) + [v_all]
    if not all(np.isfinite(x) for x in vals):
        raise AssertionError(f"non-finite value in efficiency check: phi={phi}, v_all={v_all}")
    gap = abs(sum(phi.values()) - v_all)
    if gap >= tol:
        raise AssertionError(f"efficiency violated: sum phi - v(all) = {gap:.3e}")
    return gap

"""End-to-end estimands.

pe_game          -- the PE game from a sampler and a model: coalition table, exact Shapley values,
                    efficiency gap (permutation Shapley for large K: pass n_perm).
nie_model_based  -- NIE of the fitted model with the TRUE mechanisms of an SCM:
                    E[f(t, M(t), W)] - E[f(t, M(t'), W)], same noise in both worlds.
nde_model_based, ate_model_based -- likewise.
oracle_pe        -- the PE game with the TRUE mechanisms (scm.OracleSampler) and the fitted model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .game import coalition_table, shapley_exact, shapley_permutation, check_efficiency
from .sampler import make_mean_do
from .scm import OracleSampler


def pe_game(sampler, f, players: dict, reference: dict, J: int | None = None, ate: float | None = None,
            n_perm: int | None = None, seed: int = 0) -> dict:
    """Coalition table of v(S) = ATE - CDE(c_S) at `reference`, Shapley values and efficiency gap."""
    mean_do = make_mean_do(sampler, f, J)
    if n_perm is None:
        tab = coalition_table(mean_do, players, reference, ate=ate)
        phi = shapley_exact(tab["v"], tab["players"])
        gap = check_efficiency(phi, tab["v"][tuple(tab["players"])])
        return {**tab, "phi": phi, "efficiency_gap": gap}
    names = sorted(players)
    ate_val = ate if ate is not None else mean_do(1, {}) - mean_do(0, {})

    def v(S):
        pins = {c: reference[c] for p in S for c in players[p]}
        return ate_val - (mean_do(1, pins) - mean_do(0, pins))
    phi, info = shapley_permutation(v, names, n_perm, seed=seed)
    return {"ate": ate_val, "phi": phi, "players": names, **info}


def _worlds(scm, J: int, seed: int, W_rows: pd.DataFrame | None = None):
    """W once (from the SCM, or resampled from given rows), and the mediators under T = 1 and T = 0
    drawn with the same random numbers (the draw order of every SCM does not depend on T)."""
    if scm.pretreatment:
        W = scm.draw_W(J, np.random.default_rng(seed)) if W_rows is None else \
            W_rows[scm.pretreatment].sample(J, replace=True, random_state=seed).reset_index(drop=True)
    else:
        W = pd.DataFrame(index=range(J))
    M1 = scm.draw_mediators(np.ones(J), W, {}, np.random.default_rng(seed + 1))
    M0 = scm.draw_mediators(np.zeros(J), W, {}, np.random.default_rng(seed + 1))
    return W, M1, M0


def _frame(W, t, M):
    return pd.concat([W, pd.DataFrame({"T": np.full(len(M), float(t))}), M], axis=1)


def nie_model_based(scm, f, J: int = 200_000, seed: int = 0, W_rows=None) -> float:
    W, M1, M0 = _worlds(scm, J, seed, W_rows)
    return float(np.mean(f(_frame(W, 1, M1))) - np.mean(f(_frame(W, 1, M0))))


def nde_model_based(scm, f, J: int = 200_000, seed: int = 0, W_rows=None) -> float:
    W, M1, M0 = _worlds(scm, J, seed, W_rows)
    return float(np.mean(f(_frame(W, 1, M0))) - np.mean(f(_frame(W, 0, M0))))


def ate_model_based(scm, f, J: int = 200_000, seed: int = 0, W_rows=None) -> float:
    W, M1, M0 = _worlds(scm, J, seed, W_rows)
    return float(np.mean(f(_frame(W, 1, M1))) - np.mean(f(_frame(W, 0, M0))))


def oracle_pe(scm, f, X_train: pd.DataFrame, players: dict, reference: dict, J: int = 4000, seed: int = 0,
              ate: float | None = None) -> dict:
    """The PE game with the true mechanisms of `scm` (and the empirical law of W from X_train)."""
    return pe_game(OracleSampler(scm, X_train, J=J, seed=seed), f, players, reference, J=J, ate=ate)

"""Causal Shapley (Heskes, Sijben, Bucur, Claassen 2020): phi_T with its direct and
indirect parts, all inputs as players, interventional value function.

Synthetic: interventional laws from pe_shap.sampler.MechanismSampler.  T, when free, follows
its natural mechanism: drawn with W when W is free (resampled rows), from P_hat(T | w) on a
cubic-spline basis of w when W is pinned.
"""
from __future__ import annotations

import itertools, math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


class CausalShapleyT:
    def __init__(self, sampler, treatment: str, pretreatment: list, J: int = 2000, seed: int = 0):
        self.s, self.t, self.w, self.J, self.seed = sampler, treatment, list(pretreatment), J, seed
        X = sampler.X
        if self.w:
            self.pT_ = make_pipeline(SplineTransformer(degree=3, n_knots=8), LogisticRegression(C=1e6, max_iter=2000)).fit(
                X[self.w].to_numpy(float), X[self.t].to_numpy(int))

    def _mean(self, f, x: dict, S: tuple, t_override=None) -> float:
        """E[f(x_S, X_{-S}) | do(X_S = x_S)], with the model's T argument replaced by
        t_override if given (for the direct part)."""
        pins = {k: x[k] for k in S if k != self.t}        # mediators AND pre-treatment inputs:
        # the sampler sets a pinned W before drawing the mediators, so a mediator
        # with W as a parent gets its law given the pinned W.
        if self.t in S:
            rows = self.s.draw(x[self.t], pins, self.J)
        else:
            # T free: natural mechanism.  draw() needs a value of T for the mediators, so
            # draw T first, then mediators given that T via two arm draws mixed by T.
            rows1, rows0 = self.s.draw(1, pins, self.J), self.s.draw(0, pins, self.J)
            if self.w and all(k in S for k in self.w):
                p1 = self.pT_.predict_proba(np.tile([x[k] for k in self.w], (self.J, 1)))[:, 1]
            elif self.w:
                p1 = self.pT_.predict_proba(rows1[self.w].to_numpy(float))[:, 1]
            else:
                p1 = np.full(self.J, float(self.s.X[self.t].mean()))
            u = np.random.default_rng(self.seed).uniform(size=self.J)
            take1 = u < p1
            rows = rows1.where(pd.Series(take1, index=rows1.index), rows0)
        if t_override is not None:
            rows[self.t] = float(t_override)
        return float(np.mean(f(rows)))

    def phi_T(self, f, x: dict, players: list) -> dict:
        """Returns {"phi": .., "direct": .., "indirect": ..} for player T at sample x.

        For each S not containing T, with weight w_S = |S|!(n-|S|-1)!/n!:
          total    += w_S [ v(S + T) - v(S) ]
          direct   += w_S [ E(f | do(x_S), T-argument := x_T) - v(S) ]   (law of X_{-S} as under do(x_S), only f's T changes)
          indirect += w_S [ v(S + T) - E(f | do(x_S), T-argument := x_T) ]
        """
        others = [p for p in players if p != self.t]
        n = len(players)
        tot = dr = ind = 0.0
        for k in range(n):
            wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(others, k):
                vS = self._mean(f, x, S)
                vST = self._mean(f, x, S + (self.t,))
                vS_t = self._mean(f, x, S, t_override=x[self.t])
                tot += wgt * (vST - vS); dr += wgt * (vS_t - vS); ind += wgt * (vST - vS_t)
        return {"phi": tot, "direct": dr, "indirect": ind}


def causal_shapley_compas(f, X_eval: pd.DataFrame, x: dict, treatment: str, age: str, mediators: list,
                          h: float = 0.4) -> dict:
    """Causal Shapley values of all inputs for the single prediction f(x) on COMPAS (EXP-7).

    Exact over the 2^5 coalitions with v(S) = E[f(x_S, X_{-S}) | do(X_S = x_S)] under the assumed graph
    (race and age associated, neither causing the other; every mediator has parents race and age; no
    mediator causes another).  The free inputs come from the rows of X_eval:
      race and age pinned   -> rows of the arm x[race] with age within h SDs of x[age], unweighted;
      race pinned, age free -> rows of the arm weighted to the pooled age law (RowPinningSampler's weights);
      age pinned, race free -> rows with age in the window, weighted to the pooled race proportions;
      neither pinned        -> all rows.
    Pinned inputs are overwritten with x's values.  Returns phi for every input, v(empty), f(x) and the
    smallest number of rows behind a coalition value.
    """
    from pe_shap.sampler import RowPinningSampler
    X = X_eval.reset_index(drop=True)
    players = [treatment, age] + list(mediators)
    B = RowPinningSampler(X, treatment, [age], list(mediators))
    T = X[treatment].to_numpy(int)
    window = (X[age] - x[age]).abs().to_numpy() <= h * float(X[age].std(ddof=1))
    p_pool = T.mean()
    p_win = T[window].mean()
    w_age = np.where(T == 1, p_pool / p_win, (1 - p_pool) / (1 - p_win))      # window rows to pooled race shares
    min_rows = [len(X)]

    def v(S):
        pins = {k: x[k] for k in S if k in mediators}
        if treatment in S and age not in S:
            min_rows.append(int((T == int(x[treatment])).sum()))
            return B.mean_do(f, int(x[treatment]), pins)
        rows = X.copy()
        keep, w = np.ones(len(X), bool), np.ones(len(X))
        if age in S:
            keep &= window
            rows[age] = x[age]
            if treatment in S:
                keep &= (T == int(x[treatment]))
            else:
                w = w_age
        if treatment in S:
            rows[treatment] = x[treatment]
        for k, val in pins.items():
            rows[k] = val
        min_rows.append(int(keep.sum()))
        if keep.sum() == 0:
            raise ValueError(f"causal_shapley_compas: no rows for coalition {S}")
        pred = np.asarray(f(rows.loc[keep]), float)
        return float(np.sum(w[keep] * pred) / np.sum(w[keep]))

    vals = {S: v(S) for k in range(len(players) + 1) for S in itertools.combinations(players, k)}
    n = len(players)
    phi = {}
    for j in players:
        others = [p for p in players if p != j]
        tot = 0.0
        for k in range(n):
            wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(others, k):
                Sj = tuple(p for p in players if p in S or p == j)
                tot += wgt * (vals[Sj] - vals[S])
        phi[j] = tot
    return {"phi": phi, "v_empty": vals[()], "f_x": vals[tuple(players)], "min_rows": min(min_rows)}

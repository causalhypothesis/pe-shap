"""Samplers for the interventional law of the model's inputs.

Both classes expose

    mean_do(f, tau, pins) -> float

an estimate of  mu(tau, S, c_S) = E[ f(X) | do(T = tau, C_S = c_S) ]  (Appendix E), where `pins` is {column: value} for the coalition S.  Every coalition value
of the paper is a difference of such means (pe_shap.game.coalition_table).

MechanismSampler  -- general DAG over the inputs; fits one regression per mediator.
RowPinningSampler -- parallel mediators; no mechanism model; needs no free mediator to be
                     a descendant of a pinned one.  Optional stratified variant for one
                     mediator-to-mediator edge with a discrete parent (graph-sensitivity
                     rows only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


class InvalidPinning(ValueError):
    """Row pinning is not valid for this coalition under the declared graph."""


class EmptyStratum(ValueError):
    """No row of the arm (or of the stratum of the stratified variant) matches the pins: the
    arm mean is undefined.  The experiment script catches it, writes null with the reason and
    the stratum counts for that split (a declared nullable leaf), and never
    substitutes a value or widens the stratum."""


def _topological(graph: dict[str, list[str]], nodes: list[str]) -> list[str]:
    """Order `nodes` so that every parent (within `nodes`) precedes its children."""
    order, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for p in graph.get(v, []):
            if p in nodes:
                visit(p)
        order.append(v)

    for v in nodes:
        visit(v)
    return order


class MechanismSampler:
    """g-formula by ancestral sampling along the declared DAG (Sampler A).

    Parameters
    ----------
    X : training rows (all model inputs; may contain other columns, ignored).
    graph : {node: [parents]} over the inputs; the treatment has no mediator parents.
    treatment : column name of T (values 0/1).
    pretreatment : columns W (resampled jointly as whole rows from all training rows).
    mediators : mediator columns; any order (sorted topologically here).
    mechanism : "ols" (main-effects least squares; correct for S1-S3, the two-chain
        design uses "logit" for binary nodes) or "hgb" (histogram gradient boosting,
        optional experiment only).  Residuals are pooled over arms and stored.
    seed : seed of the common random numbers (row indices for W, residual indices per
        mediator), drawn once per instance and reused for both arms, all coalitions and
        all reference rows (common random numbers).
    """

    def __init__(self, X: pd.DataFrame, graph: dict, treatment: str, pretreatment: list,
                 mediators: list, mechanism: str = "ols", seed: int = 0):
        self.X = X.reset_index(drop=True)
        self.graph = graph
        self.t = treatment
        self.w = list(pretreatment)
        self.m = _topological(graph, list(mediators))
        self.mechanism = mechanism
        self.seed = seed
        self.fitted_ = False

    # -- fit -------------------------------------------------------------------------
    def fit(self) -> "MechanismSampler":
        self.models_, self.resid_, self.kind_ = {}, {}, {}
        for v in self.m:
            pa = list(self.graph[v])
            Z, y = self.X[pa].to_numpy(float), self.X[v].to_numpy(float)
            if self.mechanism == "logit" or set(np.unique(y)) <= {0.0, 1.0}:
                mdl = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y)
                self.models_[v], self.kind_[v] = mdl, "bernoulli"
            elif self.mechanism == "hgb":
                from sklearn.ensemble import HistGradientBoostingRegressor
                mdl = HistGradientBoostingRegressor(random_state=self.seed).fit(Z, y)
                self.models_[v], self.kind_[v] = mdl, "additive"
                self.resid_[v] = y - mdl.predict(Z)
            else:
                mdl = LinearRegression().fit(Z, y)
                self.models_[v], self.kind_[v] = mdl, "additive"
                self.resid_[v] = y - mdl.predict(Z)
        self.fitted_ = True
        return self

    # -- common random numbers ---------------------------------------------------------
    def _crn(self, J: int):
        """Row indices for W and, per mediator, residual indices or uniforms; cached by J."""
        if getattr(self, "_crn_cache", {}).get("J") == J:
            return self._crn_cache
        rng = np.random.default_rng(self.seed)
        n = len(self.X)
        cache = {"J": J, "rows": rng.integers(0, n, J),
                 "resid": {v: rng.integers(0, n, J) for v in self.m},
                 "unif": {v: rng.uniform(size=J) for v in self.m}}
        self._crn_cache = cache
        return cache

    # -- draw ------------------------------------------------------------------------
    def draw(self, tau: int, pins: dict, J: int) -> pd.DataFrame:
        """J rows from the estimate of P(inputs | do(T=tau, C_S=c_S)).

        pins may name mediators and pre-treatment inputs; J defaults to 2000.
        (i) W: whole rows resampled from all training rows (their joint law is not
        changed by interventions on T or the mediators), then any pinned W set; (ii) T = tau; (iii) each
        mediator in topological order: its pin if pinned, else fitted mean at the
        current parent values plus a pooled residual (additive) or a Bernoulli draw
        (binary).  Common random numbers across arms, coalitions and reference rows.
        """
        assert self.fitted_, "call fit() first"
        if J is None:
            J = 2000
        crn = self._crn(J)
        out = self.X.loc[crn["rows"], self.w].reset_index(drop=True) if self.w else pd.DataFrame(index=range(J))
        for w in self.w:            # a pinned pre-treatment input is set BEFORE the mediators are drawn
            if w in pins:           # (Causal Shapley pins W; the PE game never does)
                out[w] = float(pins[w])
        out[self.t] = float(tau)
        for v in self.m:
            if v in pins:
                out[v] = float(pins[v])
                continue
            Z = out[list(self.graph[v])].to_numpy(float)
            if self.kind_[v] == "bernoulli":
                p = self.models_[v].predict_proba(Z)[:, 1]
                out[v] = (crn["unif"][v] < p).astype(float)
            else:
                out[v] = self.models_[v].predict(Z) + self.resid_[v][crn["resid"][v]]
        return out

    def mean_do(self, f, tau: int, pins: dict, J: int = 2000) -> float:
        """E[f(X) | do(T=tau, C_S=c_S)] by Monte Carlo with J draws."""
        return float(np.mean(f(self.draw(tau, pins, J))))


class RowPinningSampler:
    """Row pinning with propensity weights (Sampler B).

        mu_hat(tau, S, c_S) = sum_{i: T_i = tau} w_i f(tau, W_i, c_S, C_{-S, i}) / sum w_i,
        w_i = 1 / P_hat(T = tau | W_i)   (1 when there are no pre-treatment columns).

    P_hat: logistic regression of T on the standardised W (main effects, scikit-learn's
    default ridge penalty, fitted on the rows of X_eval), clipped to [clip, 1-clip].
    The weighted arm then has the pooled law of W; the free mediators keep their law
    given (T = tau, W).

    Validity (checked per coalition by `check`): no free mediator may be a descendant
    of a pinned one under the declared edges among mediators.  Dependence among the
    mediators given (T, W) does not by itself invalidate row pinning; an edge out of a
    pinned mediator does.

    strat_edge = (parent, child): stratified variant for ONE mediator-to-mediator edge
    with a discrete parent.  When the parent is pinned and the child is free, the sum is
    restricted to the arm's rows whose parent equals the pin and the weights are
    multiplied by 1 / P_hat(parent = pin | T = tau, W) (logistic regression on W within
    the arm, clipped as above).  The smallest stratum size is recorded in
    `min_stratum_`.  Exact only when that edge is the sole dependence between the
    pinned parent and the free mediators given (T, W) (EXP-6).
    """

    def __init__(self, X_eval: pd.DataFrame, treatment: str, pretreatment: list, mediators: list,
                 mediator_edges: list[tuple] = (), strat_edge: tuple | None = None, clip: float = 1e-3):
        self.X = X_eval.reset_index(drop=True)
        self.t, self.w, self.m = treatment, list(pretreatment), list(mediators)
        self.edges = [tuple(e) for e in mediator_edges]
        self.strat = tuple(strat_edge) if strat_edge else None
        self.clip = clip
        self.min_stratum_ = None
        self._fit_propensity()

    def _fit_propensity(self):
        T = self.X[self.t].to_numpy(int)
        if not self.w:
            self.p1_ = np.full(len(self.X), T.mean())
            self.clipped_share_ = 0.0
            return
        Z = StandardScaler().fit_transform(self.X[self.w].to_numpy(float))
        p = LogisticRegression(max_iter=1000).fit(Z, T).predict_proba(Z)[:, 1]
        self.clipped_share_ = float(np.mean((p < self.clip) | (p > 1 - self.clip)))
        self.p1_ = np.clip(p, self.clip, 1 - self.clip)
        self._Z = Z

    def _descendants(self, v):
        out, stack = set(), [v]
        while stack:
            u = stack.pop()
            for a, b in self.edges:
                if a == u and b not in out:
                    out.add(b)
                    stack.append(b)
        return out

    def check(self, pins: dict) -> None:
        """Raise InvalidPinning if a free mediator descends from a pinned one, unless the
        only such edge is `strat_edge` with its parent pinned and its child free."""
        pinned = [v for v in self.m if v in pins]
        for v in pinned:
            bad = [d for d in self._descendants(v) if d not in pins]
            if bad and not (self.strat and v == self.strat[0] and bad == [self.strat[1]]):
                raise InvalidPinning(f"free mediator(s) {bad} descend from pinned {v}")

    def mean_do(self, f, tau: int, pins: dict) -> float:
        """Weighted arm mean of f with the pinned columns overwritten by their pins."""
        self.check(pins)
        arm = (self.X[self.t].to_numpy(int) == int(tau))
        p = self.p1_ if tau == 1 else 1 - self.p1_
        w = 1.0 / p
        rows = self.X.copy()
        for c, val in pins.items():
            rows[c] = val
        rows[self.t] = float(tau)
        keep = arm.copy()
        if self.strat and self.strat[0] in pins and self.strat[1] not in pins:
            parent, pin = self.strat[0], pins[self.strat[0]]
            in_stratum = (self.X[parent].to_numpy() == pin)
            # P(parent = pin | T = tau, W), logistic regression on W within the arm
            y = in_stratum[arm].astype(int)
            if self.w and 0 < y.sum() < len(y):
                q = LogisticRegression(max_iter=1000).fit(self._Z[arm], y).predict_proba(self._Z)[:, 1]
            else:
                q = np.full(len(self.X), max(y.mean(), self.clip))
            q = np.clip(q, self.clip, 1 - self.clip)
            w = w / q
            keep = arm & in_stratum
            n_str = int(keep.sum())
            self.min_stratum_ = n_str if self.min_stratum_ is None else min(self.min_stratum_, n_str)
        if int(keep.sum()) == 0:          # empty arm or stratum: undefined, before any prediction is attempted
            raise EmptyStratum(f"no rows for arm {tau} with pins {pins} (stratified: {bool(self.strat)})")
        pred = np.asarray(f(rows.loc[keep]), float)
        return float(np.sum(w[keep] * pred) / np.sum(w[keep]))


def make_mean_do(sampler, f, J: int | None = None):
    """Close over a sampler and a model: mean_do(tau, pins) for pe_shap.game.

    J is the draw budget per arm and coalition for the mechanism sampler (default 2000;
    the experiment scripts pass their protocol's value explicitly and never rely on the
    default); the row-pinning sampler uses every row and ignores J."""
    if isinstance(sampler, MechanismSampler):
        J_eff = 2000 if J is None else int(J)
        return lambda tau, pins: sampler.mean_do(f, tau, pins, J_eff)
    return lambda tau, pins: sampler.mean_do(f, tau, pins)

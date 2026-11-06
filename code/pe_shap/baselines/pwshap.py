"""PW-SHAP effects from their definitions (Ter-Minassian et al., ICML 2023).

Sign convention: PW-SHAP's own.  The tables print -Psi where a table is
sign-aligned with the PE game, and say so in the caption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


class PWShapRegression:
    """Conditional contrasts on the synthetic designs from within-arm laws.

    X_train: rows with all inputs; covariates in the model's order; `pretreatment` names the
    covariates that are pre-treatment inputs (S3: ["W"]).  Per arm tau the class stores the
    mean vector and covariance of the covariates (continuous case) and the arm's rows (binary
    case and the empirical draw of pre-treatment inputs).  No graph is used.

    _cond_mean draws the FREE covariates jointly given the fixed ones:
      * a free PRE-TREATMENT input is drawn from its empirical within-arm distribution (rows of
        the arm, with replacement) — not from a Gaussian: on S3 the within-arm law of W is a
        tilted normal with no mass above 0.8 in the treated arm (P(T=1|W)=clip(0.8-W,0,1)),
        and a moment-matched Gaussian targets a different contrast for a predictor that is not
        additive in W;
      * the free continuous MEDIATORS are then drawn from the Gaussian conditional of the
        within-arm joint law given the fixed covariates AND the drawn pre-treatment values
        (exact for the within-arm mediator laws of S1-S3, whose mediators are jointly Gaussian
        given T and, on S3, independent of W given T; the regression estimate of PW-SHAP's
        contrast for a fitted model);
      * all free covariates binary     -> empirical joint frequency within the arm given the
        fixed covariates (exact for binary data; the two-chain design);
      * mixed binary/continuous free mediators, or a binary design with a free pre-treatment
        input -> NotImplementedError (use donors / a frequency table).
    A sequential one-at-a-time sweep from the reference is NOT a joint draw
    (on S1 it gives an arm shift of 56/149 instead of 0.56).
    """

    def __init__(self, X_train: pd.DataFrame, treatment: str, covariates: list, pretreatment: list = (),
                 J: int = 20_000, seed: int = 0):
        self.t, self.c, self.J, self.seed = treatment, list(covariates), J, seed
        self.w = [k for k in self.c if k in set(pretreatment)]
        self.binary_ = {j: set(np.unique(X_train[j].to_numpy(float))) <= {0.0, 1.0} for j in self.c}
        self.mu_, self.cov_, self.arm_ = {}, {}, {}
        for tau in (0, 1):
            arm = X_train[X_train[self.t] == tau][self.c].reset_index(drop=True)
            A = arm.to_numpy(float)
            self.arm_[tau] = arm
            self.mu_[tau] = A.mean(axis=0)
            self.cov_[tau] = np.atleast_2d(np.cov(A, rowvar=False, ddof=1))

    def _cond_mean(self, f, tau, c: dict, free: list) -> float:
        """E[f | T=tau, C_{-free} = c_{-free}] with the free covariates drawn JOINTLY from
        their within-arm conditional law given the fixed ones (see the class docstring)."""
        free = list(free)
        fixed = [k for k in self.c if k not in free]
        free_w = [k for k in free if k in self.w]          # free pre-treatment inputs: empirical draw
        free_m = [k for k in free if k not in self.w]      # free mediators
        rng = np.random.default_rng(self.seed)
        rows = pd.DataFrame(index=range(self.J))
        for k in fixed:
            rows[k] = float(c[k])
        rows[self.t] = float(tau)
        arm = self.arm_[tau]
        if free_w:
            if any(self.binary_[j] for j in free_m):
                raise NotImplementedError("free pre-treatment input with binary free mediators: not supported")
            idx = rng.integers(0, len(arm), self.J)
            for k in free_w:
                rows[k] = arm[k].to_numpy(float)[idx]
        if not free_m:
            pass
        elif all(self.binary_[j] for j in free_m):
            mask = np.ones(len(arm), dtype=bool)
            for k in fixed:
                mask &= arm[k].to_numpy(float) == float(c[k])
            if mask.sum() == 0:
                raise ValueError(f"no rows in arm {tau} with the fixed covariates {dict((k, c[k]) for k in fixed)}")
            pool = arm.loc[mask, free_m].to_numpy(float)
            idx = rng.integers(0, len(pool), self.J)
            for i, j in enumerate(free_m):
                rows[j] = pool[idx, i]
        elif not any(self.binary_[j] for j in free_m):
            cond = fixed + free_w                            # condition on the fixed values and the drawn W
            iF = [self.c.index(j) for j in free_m]
            iX = [self.c.index(k) for k in cond]
            mu, S = self.mu_[tau], self.cov_[tau]
            if iX:
                Sxx = S[np.ix_(iX, iX)]; Sfx = S[np.ix_(iF, iX)]
                A = np.linalg.solve(Sxx, Sfx.T).T                      # Sfx Sxx^{-1}
                X = rows[cond].to_numpy(float)                          # J x |cond|; constant columns for fixed values
                M = mu[iF] + (X - mu[iX]) @ A.T                         # per-row conditional mean
                V = S[np.ix_(iF, iF)] - A @ Sfx.T
            else:
                M, V = np.tile(mu[iF], (self.J, 1)), S[np.ix_(iF, iF)]
            V = (V + V.T) / 2
            Z = rng.multivariate_normal(np.zeros(len(iF)), V, size=self.J, method="cholesky" if len(iF) == 1 else "svd")
            draws = M + Z
            for i, j in enumerate(free_m):
                rows[j] = draws[:, i]
        else:
            raise NotImplementedError("mixed binary/continuous free block: not supported by PWShapRegression")
        return float(np.mean(f(rows)))

    def delta(self, f, c: dict, free: list = ()) -> float:
        """Delta_t(c_S): conditional contrast given all covariates except `free`."""
        if not free:
            row = pd.DataFrame({k: [float(c[k])] for k in self.c})
            r1, r0 = row.copy(), row.copy()
            r1[self.t], r0[self.t] = 1.0, 0.0
            return float(f(r1)[0] - f(r0)[0])
        return self._cond_mean(f, 1, c, list(free)) - self._cond_mean(f, 0, c, list(free))

    def psi(self, f, c: dict, j: str) -> float:
        """Def. 3.2: Delta_t(c) - Delta_t(c_{-j})."""
        return self.delta(f, c) - self.delta(f, c, free=[j])

    def psi_block(self, f, c: dict, block: list) -> float:
        """Delta_t(c) - Delta_t(c_{-B})."""
        return self.delta(f, c) - self.delta(f, c, free=list(block))

    def psi_alt(self, f, c: dict, j: str) -> float:
        """Suppl. F.2 alternative: Delta_t(c_j) - Delta_t(c_empty)."""
        free_all = [k for k in self.c]
        return self.delta(f, c, free=[k for k in self.c if k != j]) - self.delta(f, c, free=free_all)


class PWShapDonors:
    """Conditional contrasts on real data from matched donor rows of each arm (COMPAS and
    German Credit).  Returns window-smoothed estimates; records donor counts.

    rule: "matched" (other mediators matched: exact for the discrete ones, +-1 for
          priors_count; age within h SDs), "matched_exact" (priors_count exact too),
          "age_only" (age window only).
    """

    def __init__(self, X_eval: pd.DataFrame, treatment: str, age: str, mediators: list,
                 h: float = 0.4, rule: str = "matched", tolerance: dict | None = None):
        self.X, self.t, self.age, self.m, self.h, self.rule = X_eval.reset_index(drop=True), treatment, age, list(mediators), h, rule
        self.tol = tolerance or {}          # e.g. {"priors_count": 1} ; {"amount": 0.4*sd} for German
        self.sd_age = float(self.X[age].std(ddof=1))
        self.counts_ = {}
        self.undefined_ = {}      # j -> {"reason", "n_donors"} for contrasts that could not be formed

    def _donors(self, tau, ref: dict, j: str) -> pd.DataFrame:
        D = self.X[self.X[self.t] == tau]
        D = D[(D[self.age] - ref[self.age]).abs() <= self.h * self.sd_age]
        if self.rule != "age_only":
            for k in self.m:
                if k == j:
                    continue
                tol = 0 if self.rule == "matched_exact" else self.tol.get(k, 0)
                D = D[(D[k] - ref[k]).abs() <= tol]
        return D

    def delta_given_all(self, f, ref: dict) -> float:
        row = pd.DataFrame({k: [ref[k]] for k in ref})
        r1, r0 = row.copy(), row.copy()
        r1[self.t], r0[self.t] = 1, 0
        return float(f(r1)[0] - f(r0)[0])

    def delta_without(self, f, ref: dict, j: str):
        """Delta_t(c_{-j}) from the donors of each arm; None (undefined) when either arm has no
        donor.  The guard runs BEFORE any prediction: no empty frame reaches the model, no
        zero is substituted, the window and the rule are not changed.  The donor counts of both arms are recorded either way."""
        donors = {tau: self._donors(tau, ref, j) for tau in (0, 1)}
        for tau in (0, 1):
            self.counts_[(j, tau)] = len(donors[tau])
        if min(len(donors[0]), len(donors[1])) == 0:
            self.undefined_[j] = {"reason": "no donors", "n_donors": {0: len(donors[0]), 1: len(donors[1])}}
            return None
        out = {}
        for tau in (0, 1):
            D = donors[tau]
            rows = pd.DataFrame({k: np.full(len(D), ref[k]) for k in ref})
            rows[self.t] = tau
            rows[j] = D[j].to_numpy()
            out[tau] = float(np.mean(f(rows)))
        return out[1] - out[0]

    def psi(self, f, ref: dict, j: str):
        """Def. 3.2 on real data; None when the contrast without j is undefined (the script
        writes null with self.undefined_[j] beside it; the leaf is declared nullable)."""
        d = self.delta_without(f, ref, j)
        return None if d is None else self.delta_given_all(f, ref) - d

    def min_donors(self, j: str) -> int:
        return min(self.counts_[(j, 0)], self.counts_[(j, 1)])

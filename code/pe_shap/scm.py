"""Synthetic designs.  Each SCM exposes
    name, treatment, pretreatment, mediators (topological), graph, inputs,
    sample(n, seed) -> DataFrame with inputs and Y,
    draw_mediators(T, W, pins, rng) -> DataFrame   (the TRUE mechanisms; used by the
        oracle sampler and by nie_model_based),
    f_true(X) -> array (the true outcome function without noise).
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import pandas as pd

SD = 0.5


class SCM:
    name = ""; treatment = "T"; pretreatment: list = []; mediators: list = []; graph: dict = {}

    @property
    def inputs(self):
        return self.pretreatment + [self.treatment] + self.mediators

    def draw_W(self, n, rng):
        return pd.DataFrame(index=range(n))

    def draw_T(self, W, rng):
        # rng.binomial, as the shipped generators (data/simulations.py): with it, sample(n, seed)
        # reproduces the submission's S1-S3 data exactly for the same seed
        return rng.binomial(1, 0.5, len(W)).astype(float)

    def draw_mediators(self, T, W, pins, rng):
        raise NotImplementedError

    def f_true(self, X):
        raise NotImplementedError

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        W = self.draw_W(n, rng)
        T = self.draw_T(W, rng)
        M = self.draw_mediators(T, W, {}, rng)
        X = pd.concat([W, pd.DataFrame({"T": T}), M], axis=1)
        X["Y"] = self.f_true(X) + rng.normal(0, SD, n)
        return X


class S1(SCM):
    name, mediators, graph = "S1", ["M1", "M2"], {"M1": ["T"], "M2": ["M1"]}

    def draw_mediators(self, T, W, pins, rng):
        n = len(T)
        M1 = np.full(n, pins["M1"]) if "M1" in pins else 0.8 * T + rng.normal(0, SD, n)
        M2 = np.full(n, pins["M2"]) if "M2" in pins else 0.7 * M1 + rng.normal(0, SD, n)
        return pd.DataFrame({"M1": M1, "M2": M2})

    def f_true(self, X):
        return 0.5 * X["T"].to_numpy() + X["M2"].to_numpy()


class S2(SCM):
    name, mediators, graph = "S2", ["M1", "M2"], {"M1": ["T"], "M2": ["T"]}

    def draw_mediators(self, T, W, pins, rng):
        n = len(T)
        M1 = np.full(n, pins["M1"]) if "M1" in pins else 0.5 * T + rng.normal(0, SD, n)
        M2 = np.full(n, pins["M2"]) if "M2" in pins else 0.7 * T + rng.normal(0, SD, n)
        return pd.DataFrame({"M1": M1, "M2": M2})

    def f_true(self, X):
        T, M1, M2 = (X[c].to_numpy() for c in ("T", "M1", "M2"))
        return 0.3 * T + 0.2 * M1 - 1.1 * M2 + 0.2 * M1 * T + 0.8 * M1 * M2


class S3(SCM):
    name, pretreatment, mediators = "S3", ["W"], ["M1", "M2"]
    graph = {"T": ["W"], "M1": ["T"], "M2": ["M1", "T"]}

    def __init__(self, law="submitted"):
        assert law in ("submitted", "clipped"); self.law = law

    def propensity(self, W):
        lo, hi = (0.0, 1.0) if self.law == "submitted" else (0.05, 0.95)
        return np.clip(0.8 - W, lo, hi)

    def draw_W(self, n, rng):
        return pd.DataFrame({"W": rng.normal(0.3, SD, n)})

    def draw_T(self, W, rng):
        return rng.binomial(1, self.propensity(W["W"].to_numpy())).astype(float)

    def draw_mediators(self, T, W, pins, rng):
        n = len(T)
        M1 = np.full(n, pins["M1"]) if "M1" in pins else 0.5 * T + rng.normal(0, SD, n)
        M2 = np.full(n, pins["M2"]) if "M2" in pins else 0.7 * M1 + 0.3 * T + rng.normal(0, SD, n)
        return pd.DataFrame({"M1": M1, "M2": M2})

    def f_true(self, X):
        W, T, M1, M2 = (X[c].to_numpy() for c in ("W", "T", "M1", "M2"))
        return W + T - 0.8 * M1 ** 2 - 0.5 * T * M1 + 0.4 * M1 * M2


class TwoChains(SCM):
    name, mediators = "two_chains", ["A1", "A2", "B1", "B2"]
    graph = {"A1": ["T"], "A2": ["A1"], "B1": ["T"], "B2": ["B1"]}

    def __init__(self, variant=False):
        self.variant = variant   # f = 0.3T + (1+T) A2 B2

    def draw_mediators(self, T, W, pins, rng):
        n = len(T)
        def bern(p, key):
            return np.full(n, float(pins[key])) if key in pins else (rng.uniform(size=n) < p).astype(float)
        A1 = bern(0.2 + 0.6 * T, "A1"); A2 = bern(0.2 + 0.6 * A1, "A2")
        B1 = bern(0.3 + 0.4 * T, "B1"); B2 = bern(0.1 + 0.5 * B1, "B2")
        return pd.DataFrame({"A1": A1, "A2": A2, "B1": B1, "B2": B2})

    def f_true(self, X):
        T, A2, B2 = (X[c].to_numpy() for c in ("T", "A2", "B2"))
        return 0.3 * T + (1 + T if self.variant else 1) * A2 * B2


class KDesign(SCM):
    """Parallel mediators with pairwise interactions; coefficients from configs/*.json."""
    name = "kdesign"

    def __init__(self, config_path):
        cfg = json.loads(pathlib.Path(config_path).read_text())
        self.alpha = np.asarray(cfg["alpha"], float); self.beta0 = float(cfg["beta0"])
        self.beta = np.asarray(cfg["beta"], float); self.gamma = {tuple(k): float(v) for k, v in cfg["gamma"]}
        self.K = len(self.alpha); self.mediators = [f"M{k+1}" for k in range(self.K)]
        self.graph = {m: ["T"] for m in self.mediators}; self.provenance = cfg.get("provenance", "")

    def draw_mediators(self, T, W, pins, rng):
        n = len(T); out = {}
        for k, m in enumerate(self.mediators):
            out[m] = np.full(n, pins[m]) if m in pins else self.alpha[k] * T + rng.normal(0, SD, n)
        return pd.DataFrame(out)

    def f_true(self, X):
        T = X["T"].to_numpy(); M = X[self.mediators].to_numpy()
        y = self.beta0 * T + M @ self.beta
        for (k, l), g in self.gamma.items():
            y = y + g * M[:, k] * M[:, l]
        return y


class OracleSampler:
    """mean_do with the TRUE mechanisms of an SCM and the empirical law of W (EXP-1c)."""

    def __init__(self, scm, X_train, J=4000, seed=0):
        self.scm, self.X, self.J, self.seed = scm, X_train.reset_index(drop=True), J, seed

    def draw(self, tau, pins, J=None):
        J = J or self.J; rng = np.random.default_rng(self.seed)
        W = self.X.loc[rng.integers(0, len(self.X), J), self.scm.pretreatment].reset_index(drop=True)
        T = np.full(J, float(tau))
        M = self.scm.draw_mediators(T, W, pins, rng)
        return pd.concat([W, pd.DataFrame({"T": T}), M], axis=1)

    def mean_do(self, f, tau, pins, J=None):
        return float(np.mean(f(self.draw(tau, pins, J))))

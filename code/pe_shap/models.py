"""Model training as shipped, and the callable wrapper every sampler expects."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier


def as_callable(est, columns, proba=False):
    """f(X: DataFrame) -> ndarray; selects `columns` in order, so the frame may hold more."""
    def f(X):
        Z = X[list(columns)].to_numpy(float)
        return est.predict_proba(Z)[:, 1] if proba else est.predict(Z)
    f.columns = list(columns)
    return f


def fit_mlp_regressor(X, y, seed=42):
    """As shipped for the synthetic benchmarks: (64, 64), max_iter=1000, random_state=42."""
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=seed).fit(X, y)


def fit_mlp_classifier(X, y, seed):
    """As shipped for COMPAS and German Credit: (100, 50), seeded by the split."""
    return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=seed).fit(X, y)

"""Oriented comprehensiveness and sufficiency with all same-group donors.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd


def _score_fn(f, x_row: pd.DataFrame):
    cls = int(f(x_row)[0] >= 0.5)
    return (lambda X: f(X) if cls == 1 else 1 - f(X)), cls


def masked_score(f, score, x: dict, mask: tuple, donors: pd.DataFrame) -> float:
    """Mean score over donors after replacing the columns in `mask` jointly by each donor's values."""
    if not mask:
        return float(score(pd.DataFrame({k: [v] for k, v in x.items()}))[0])
    rows = pd.DataFrame({k: np.full(len(donors), v) for k, v in x.items()})
    for c in mask:
        rows[c] = donors[c].to_numpy()
    return float(np.mean(score(rows)))


def faithfulness(f, x: dict, mediators: list, ranking: list, donors: pd.DataFrame) -> dict:
    """All subset scores plus comprehensiveness/sufficiency at k=1..K for `ranking` and the
    random baseline.  Returns a dict with keys: p_hat, cls, subset (mask -> mean score),
    comp (k -> value), suff (k -> value), comp_random, suff_random."""
    x_row = pd.DataFrame({k: [v] for k, v in x.items()})
    score, cls = _score_fn(f, x_row)
    s0 = float(score(x_row)[0])
    K = len(mediators)
    subset = {}
    for k in range(K + 1):
        for B in itertools.combinations(sorted(mediators), k):
            subset[B] = masked_score(f, score, x, B, donors)

    def comp_suff(order):
        comp, suff = {}, {}
        for k in range(1, K + 1):
            top = tuple(sorted(order[:k])); rest = tuple(sorted(order[k:]))
            comp[k] = s0 - subset[top]
            suff[k] = s0 - subset[rest]
        return comp, suff

    comp, suff = comp_suff(list(ranking))
    rnd = [comp_suff(list(p)) for p in itertools.permutations(mediators)]
    comp_r = {k: float(np.mean([c[k] for c, _ in rnd])) for k in range(1, K + 1)}
    suff_r = {k: float(np.mean([s[k] for _, s in rnd])) for k in range(1, K + 1)}
    return {"p_hat": float(f(x_row)[0]), "cls": cls, "subset": {"+".join(B): v for B, v in subset.items()},
            "comp": comp, "suff": suff, "comp_random": comp_r, "suff_random": suff_r}


def rank_by_abs(values: dict) -> list:
    """Ranking by |value|, descending; ties broken lexicographically by name."""
    return [k for k, _ in sorted(values.items(), key=lambda kv: (-abs(kv[1]), kv[0]))]

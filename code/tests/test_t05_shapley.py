"""T5: shapley_exact against enumeration of all orderings (K <= 6); shapley_permutation on the stated
K = 8 game with the pre-specified variance-based criterion; check_efficiency."""
import itertools, math
import numpy as np, pytest
from pe_shap.game import shapley_exact, shapley_permutation, check_efficiency


def _game(K, rng):
    """Coalitions in size-then-lexicographic order, values i.i.d. U[-1, 1], v(empty) = 0."""
    names = [f"M{k+1}" for k in range(K)]
    coal = [S for k in range(K + 1) for S in itertools.combinations(names, k)]
    vals = [0.0] + list(rng.uniform(-1, 1, len(coal) - 1))
    return names, {tuple(sorted(S)): x for S, x in zip(coal, vals)}


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_exact_equals_all_orderings(K):
    names, v = _game(K, np.random.default_rng(0))
    phi = shapley_exact(v, names)
    acc = dict.fromkeys(names, 0.0)
    for order in itertools.permutations(names):
        S = []
        for p in order:
            before = v[tuple(sorted(S))]; S.append(p)
            acc[p] += v[tuple(sorted(S))] - before
    n = math.factorial(K)
    assert all(abs(phi[p] - acc[p] / n) <= 1e-12 for p in names)
    check_efficiency(phi, v[tuple(names)], tol=1e-12)


def test_permutation_estimator_k8():
    names, v = _game(8, np.random.default_rng(0))
    K, P = 8, 2000
    phi = shapley_exact(v, names)
    se = {}
    for j in names:                       # exact Var of j's increment over a uniform random ordering
        others = [p for p in names if p != j]; m2 = 0.0
        for k in range(K):
            w = math.factorial(k) * math.factorial(K - k - 1) / math.factorial(K)
            for S in itertools.combinations(others, k):
                m2 += w * (v[tuple(sorted(S + (j,)))] - v[tuple(sorted(S))]) ** 2
        se[j] = math.sqrt((m2 - phi[j] ** 2) / P)
    assert 0.015 < min(se.values()) and max(se.values()) < 0.020          # 0.0159-0.0195 on this game
    est, info = shapley_permutation(lambda S: v[tuple(sorted(S))], names, P, seed=0)
    ratio = max(abs(est[j] - phi[j]) / se[j] for j in names)
    print(f"T5: SE {min(se.values()):.4f}-{max(se.values()):.4f}; achieved max ratio {ratio:.2f} (criterion 3.23)")
    assert ratio <= 3.23
    check_efficiency(est, v[tuple(names)], tol=1e-10)                      # every ordering sums to v(all)


def test_check_efficiency_raises_on_nan():
    check_efficiency({"a": 0.2, "b": 0.3}, 0.5)
    with pytest.raises(AssertionError):
        check_efficiency({"a": float("nan"), "b": 0.3}, 0.5)
    with pytest.raises(AssertionError):
        check_efficiency({"a": 0.2, "b": 0.3}, 0.6)

"""T7: the data assertions (both loaders) and the person-disjoint split.
The data files are not in the repository (data/README.md says where they come from); a missing
file fails here, and run_tests.py leaves this test out when they are absent."""
import pathlib
from pe_shap.data import load_compas, load_german, split

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def test_compas_loader_and_splits():
    c = load_compas(DATA / "compas-scores-two-years.csv")         # the loader asserts the cohort
    for seed in range(20):
        tr, te = split(c, seed, 0.3, "id")                        # asserts person-disjointness
        assert len(tr) + len(te) == len(c)


def test_german_loader():
    g = load_german(DATA / "german_credit.csv")                    # asserts counts and codes
    assert set(g.columns) == {"female", "age", "savings", "amount", "y"}

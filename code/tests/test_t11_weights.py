"""T11: RowPinningSampler with propensity weights on a design where W affects T and the mediators;
both arm means of every coalition against the exact values; the unweighted estimator as negative
control must miss the empty coalition by more than 0.05."""
import pytest
from pe_shap.sampler import RowPinningSampler
from tests.conftest import PROBES, N_BIG, t11_sample, t11_f, t11_mu

COALS = ((), ("M1",), ("M2",), ("M1", "M2"))


@pytest.fixture(scope="module")
def X11():
    return t11_sample(N_BIG, seed=21)


def test_weighted_arm_means(X11):
    B = RowPinningSampler(X11, "T", ["W"], ["M1", "M2"])
    for m1, m2 in PROBES:
        for S in COALS:
            pins = {k: v for k, v in (("M1", m1), ("M2", m2)) if k in S}
            for tau in (1, 0):
                assert abs(B.mean_do(t11_f, tau, pins) - t11_mu(tau, pins)) <= 0.01, (m1, m2, S, tau)


def test_unweighted_misses(X11):
    U = RowPinningSampler(X11, "T", [], ["M1", "M2"])          # no pre-treatment columns: no weights
    for tau in (1, 0):
        assert abs(U.mean_do(t11_f, tau, {}) - t11_mu(tau, {})) > 0.05, tau

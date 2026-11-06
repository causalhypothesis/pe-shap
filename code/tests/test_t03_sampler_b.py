"""T3: RowPinningSampler equals MechanismSampler on S2; it refuses {M1} on S1 and S3 when the
mediator edge is declared."""
import pytest
from pe_shap.sampler import MechanismSampler, RowPinningSampler, InvalidPinning, make_mean_do
from pe_shap.game import coalition_table, shapley_exact
from tests.conftest import PROBES

PLAYERS = {"M1": ["M1"], "M2": ["M2"]}


@pytest.mark.parametrize("probe", PROBES)
def test_row_pinning_equals_mechanism_on_s2(big, probe):
    scm, X, f = big["S2"]
    ref = {"M1": probe[0], "M2": probe[1]}
    A = MechanismSampler(X, scm.graph, "T", [], scm.mediators, seed=7).fit()
    B = RowPinningSampler(X, "T", [], scm.mediators)
    ta = coalition_table(make_mean_do(A, f, 20_000), PLAYERS, ref)
    tb = coalition_table(make_mean_do(B, f), PLAYERS, ref)
    for S in ta["mu"]:
        assert abs(ta["mu"][S][0] - tb["mu"][S][0]) <= 0.01 and abs(ta["mu"][S][1] - tb["mu"][S][1]) <= 0.01, S
        assert abs(ta["v"][S] - tb["v"][S]) <= 0.01, S
    pa, pb = shapley_exact(ta["v"], ta["players"]), shapley_exact(tb["v"], tb["players"])
    assert all(abs(pa[k] - pb[k]) <= 0.01 for k in pa)


@pytest.mark.parametrize("name", ["S1", "S3"])
def test_row_pinning_refuses_m1_under_the_chain(big, name):
    scm, X, f = big[name]
    B = RowPinningSampler(X, "T", scm.pretreatment, scm.mediators, mediator_edges=[("M1", "M2")])
    with pytest.raises(InvalidPinning):
        B.mean_do(f, 1, {"M1": 0.0})
    B.mean_do(f, 1, {"M2": 0.0}); B.mean_do(f, 1, {"M1": 0.0, "M2": 0.0})    # these stay valid

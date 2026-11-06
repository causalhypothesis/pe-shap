"""T2: MechanismSampler with the true f against closed_forms at the five probes (arm means first)."""
import numpy as np, pytest
from pe_shap.sampler import MechanismSampler, make_mean_do
from pe_shap.game import coalition_table, shapley_exact, check_efficiency
from pe_shap import closed_forms as cf
from tests.conftest import PROBES

PLAYERS = {"M1": ["M1"], "M2": ["M2"]}
EXACT = {"S1": lambda m1, m2: cf.s1(m1, m2), "S2": cf.s2, "S3": lambda m1, m2: cf.s3(m1, m2, law="submitted"),
         "S3c": lambda m1, m2: cf.s3(m1, m2, law="clipped")}
COALS = (((), "0"), (("M1",), "1"), (("M2",), "2"), (("M1", "M2"), "12"))


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S3c"])
@pytest.mark.parametrize("probe", PROBES)
def test_sampler_a_matches_closed_forms(big, name, probe):
    scm, X, f = big[name]
    A = MechanismSampler(X, scm.graph, "T", scm.pretreatment, scm.mediators, seed=7).fit()
    tab = coalition_table(make_mean_do(A, f, 20_000), PLAYERS, {"M1": probe[0], "M2": probe[1]})
    ex = EXACT[name](*probe)
    # ARM MEANS first: they carry the residual-variance terms that every contrast cancels
    # (on S3 the empty coalition is (0.17, 0.85) with the
    # residual term and (0.30, 0.98) without it, while the ATE is 0.68 either way)
    for S, key in COALS:
        mu_t, mu_tprime = tab["mu"][S]
        assert abs(mu_t - ex["mu"][key]["t"]) <= 0.01, (name, probe, S, "arm mean t")
        assert abs(mu_tprime - ex["mu"][key]["tprime"]) <= 0.01, (name, probe, S, "arm mean t'")
    assert abs(tab["ate"] - ex["ate"]) <= 0.01
    for S, key in COALS[1:]:
        assert abs(tab["v"][S] - ex["lam"][key]) <= 0.01, (name, probe, S)
    phi = shapley_exact(tab["v"], tab["players"]); check_efficiency(phi, tab["v"][("M1", "M2")])
    assert abs(phi["M1"] - ex["phi"][0]) <= 0.01 and abs(phi["M2"] - ex["phi"][1]) <= 0.01

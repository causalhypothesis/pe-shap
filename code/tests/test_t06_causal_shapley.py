"""T6: Causal Shapley with the true f, direct and indirect parts of T against closed_forms at the same
reference for both numbers (S1, S2 at the five probes, S3 at pinned W under both laws), and the S3
means of closed_forms against the average of its pointwise formulas."""
import numpy as np, pytest
from pe_shap.sampler import MechanismSampler
from pe_shap.baselines.causal_shapley import CausalShapleyT
from pe_shap import closed_forms as cf
from tests.conftest import PROBES

TOL = 0.005
J = 200_000


def _cs(big, name):
    scm, X, f = big[name]
    A = MechanismSampler(X, scm.graph, "T", scm.pretreatment, scm.mediators, seed=4).fit()
    return CausalShapleyT(A, "T", scm.pretreatment, J=J, seed=5), f, scm


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_s1_s2_pointwise(big, name):
    cs, f, _ = _cs(big, name)
    for (m1, m2) in PROBES:
        ex = cf.s1(m1, m2) if name == "S1" else cf.s2(m1, m2)
        for t in (0, 1):
            r = cs.phi_T(f, {"T": t, "M1": m1, "M2": m2}, ["T", "M1", "M2"])
            assert abs(r["direct"] - ex["cs_dir"][t]) <= TOL, (name, m1, m2, t, "direct", r)
            assert abs(r["indirect"] - ex["cs_ind"][t]) <= TOL, (name, m1, m2, t, "indirect", r)


def test_named_targets():
    """Named targets, from one reference each."""
    assert np.allclose([cf.s1()["cs_dir"][0], cf.s1()["cs_ind"][0]], [-0.25, -0.0933], atol=5e-5)
    assert np.allclose([cf.s2(0, 0)["cs_dir"][0], cf.s2(0, 0)["cs_ind"][0]], [-0.175, 0.1208], atol=5e-5)
    assert np.allclose([cf.s2(0, 0)["cs_dir"][1], cf.s2(0, 0)["cs_ind"][1]], [0.15, -0.0958], atol=5e-5)
    assert np.allclose([cf.s2(0.5, 0.7)["cs_dir"][1], cf.s2(0.5, 0.7)["cs_ind"][1]], [0.175, -0.0492], atol=5e-5)


class _TrueP:
    """P(T = 1 | W) of the SCM in place of the fitted spline-logistic model."""
    def __init__(self, scm):
        self.scm = scm

    def predict_proba(self, Z):
        p = self.scm.propensity(np.asarray(Z, float)[:, 0])
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("propensity,tol", [("true", TOL), ("fitted", 0.01)])
@pytest.mark.parametrize("name,law", [("S3", "submitted"), ("S3c", "clipped")])
def test_s3_pointwise_at_pinned_w(big, name, law, propensity, tol):
    """With the true P(T | W) the code is checked at 0.005.  With the fitted cubic-spline logistic model
    the direct part also carries that model's error near the kinks of p(w) = clip(0.8 - w) (fitted 0.930
    against 0.9 at w = -0.1; 0.184 against 0.2 at w = 0.6 under the clipped law), up to 0.006 here."""
    cs, f, scm = _cs(big, name)
    if propensity == "true":
        cs.pT_ = _TrueP(scm)
    for w, (m1, m2) in zip((0.3, 0.6, -0.1, 0.9, 0.3), PROBES):
        ex = cf.s3(m1, m2, w=w, law=law)
        for t in (0, 1):
            r = cs.phi_T(f, {"W": w, "T": t, "M1": m1, "M2": m2}, ["W", "T", "M1", "M2"])
            assert abs(r["direct"] - ex["cs_dir"][t]) <= tol, (law, w, m1, m2, t, "direct", r)
            assert abs(r["indirect"] - ex["cs_ind"][t]) <= tol, (law, w, m1, m2, t, "indirect", r)


@pytest.mark.parametrize("name,law,e", [("S3", "submitted", 0.1210), ("S3c", "clipped", 0.1372)])
def test_s3_means(big, name, law, e):
    _, X, _ = big[name]
    means = cf.s3_cs_means(law)
    assert abs(means["e_p1mp"] - e) <= 5e-5
    rows = X.sample(40_000, random_state=0)
    for t, grp in ((0, "untreated"), (1, "treated")):
        R = rows[rows["T"] == t]
        d = np.mean([cf.s3(m1, m2, w=w, law=law)["cs_dir"][t] for w, m1, m2 in zip(R["W"], R["M1"], R["M2"])])
        i = np.mean([cf.s3(m1, m2, w=w, law=law)["cs_ind"][t] for w, m1, m2 in zip(R["W"], R["M1"], R["M2"])])
        assert abs(d - means[grp]["dir"]) <= TOL and abs(i - means[grp]["ind"]) <= TOL, (law, grp, d, i, means[grp])

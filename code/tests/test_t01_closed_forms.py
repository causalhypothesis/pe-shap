"""T1: closed_forms identities (exact, 1e-12): efficiency at 20 random pins, S1 symmetry,
E_{t'}[lambda_all] = NIE on S1-S3, and the two-chain rows of tab:two-chains."""
import numpy as np, pytest
from pe_shap import closed_forms as cf

TOL = 1e-12
rng = np.random.default_rng(0)
PINS = [tuple(p) for p in rng.uniform(-2, 2, (20, 2))]
WS = list(rng.uniform(-0.5, 1.1, 20))
DESIGNS = {"S1": lambda m1, m2, w: cf.s1(m1, m2), "S2": lambda m1, m2, w: cf.s2(m1, m2),
           "S3": lambda m1, m2, w: cf.s3(m1, m2, w, "submitted"), "S3c": lambda m1, m2, w: cf.s3(m1, m2, w, "clipped")}


@pytest.mark.parametrize("name", DESIGNS)
def test_efficiency_at_random_pins(name):
    for (m1, m2), w in zip(PINS, WS):
        g = DESIGNS[name](float(m1), float(m2), float(w))
        assert abs(g["phi"][0] + g["phi"][1] - g["lam"]["12"]) <= TOL, (name, m1, m2)


def test_s1_symmetric_and_s4_efficiency():
    for m1, m2 in PINS:
        g = cf.s1(float(m1), float(m2))
        assert abs(g["phi"][0] - g["phi"][1]) <= TOL
    s4 = cf.s4()
    assert abs(sum(s4["phi"]) + s4["block"]) <= TOL            # sum phi = lambda_all = -block (own sign)


# Under T = t' = 0 the mediators are jointly normal with mean 0: S1 and S3 M1 = U1, M2 = 0.7 M1 + U2;
# S2 independent.  lambda_all is a polynomial of degree <= 2 in the pins, so a 3-point Gauss-Hermite rule
# per dimension (exact to degree 5) gives its expectation exactly.
CHOL = {"S1": np.array([[0.5, 0.0], [0.35, 0.5]]), "S2": np.array([[0.5, 0.0], [0.0, 0.5]]),
        "S3": np.array([[0.5, 0.0], [0.35, 0.5]]), "S3c": np.array([[0.5, 0.0], [0.35, 0.5]])}


@pytest.mark.parametrize("name", DESIGNS)
def test_expected_lambda_all_is_nie(name):
    x, wq = np.polynomial.hermite_e.hermegauss(3)
    wq = wq / wq.sum()
    e = 0.0
    for i in range(3):
        for j in range(3):
            m1, m2 = CHOL[name] @ np.array([x[i], x[j]])
            e += wq[i] * wq[j] * DESIGNS[name](float(m1), float(m2), 0.3)["lam"]["12"]
    assert abs(e - DESIGNS[name](0.0, 0.0, 0.3)["nie"]) <= TOL, (name, e)


def test_two_chain_rows():
    """tab:two-chains as printed (three decimals)."""
    tc, tv = cf.two_chains(), cf.two_chains(variant=True)
    rows = {("untreated", "pw_A"): 0.090, ("untreated", "pw_B"): 0.064, ("untreated", "pw_sum"): 0.154,
            ("treated", "pw_A"): 0.162, ("treated", "pw_B"): 0.136, ("treated", "pw_sum"): 0.298,
            ("both", "pw_A"): 0.126, ("both", "pw_B"): 0.100, ("both", "pe_A"): 0.126, ("both", "pe_B"): 0.100,
            ("both", "joint"): 0.226}
    for (g, k), printed in rows.items():
        assert f"{tc[g][k]:.3f}" == f"{printed:.3f}", (g, k, tc[g][k])
    for g in ("untreated", "treated"):
        assert abs(tc[g]["pe_A"] + tc[g]["pe_B"] - tc[g]["joint"]) <= TOL
    for g, printed in (("untreated", -0.144), ("treated", 0.072), ("both", -0.036)):
        assert f"{tv[g]['gap']:.3f}" == f"{printed:.3f}", (g, tv[g]["gap"])

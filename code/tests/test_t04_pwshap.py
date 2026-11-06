"""T4: PW-SHAP (PWShapRegression) with the true f against closed_forms on S1-S3 at the five probes,
the S1 block (the joint-conditional probe) and order invariance, the identity with PE-SHAP's last
position on S1 and S2 and its failure on S3, the alternatives on S4, the binary two chains, and the empirical draw of a free W."""
import itertools
import numpy as np, pytest
from pe_shap.baselines.pwshap import PWShapRegression
from pe_shap.scm import TwoChains
from pe_shap import closed_forms as cf, _two_chains_exact as tce
from tests.conftest import PROBES
from tests.test_t02_sampler_a import EXACT

TOL = 0.01


def _pw(big, name):
    scm, X, f = big[name]
    return PWShapRegression(X, "T", scm.pretreatment + scm.mediators, pretreatment=scm.pretreatment, seed=1), f


def _ref(name, probe):
    return {"M1": probe[0], "M2": probe[1], **({"W": 0.3} if name.startswith("S3") else {})}


@pytest.mark.parametrize("name", ["S1", "S2", "S3"])
@pytest.mark.parametrize("probe", PROBES)
def test_psi_and_block(big, name, probe):
    pw, f = _pw(big, name)
    c, ex = _ref(name, probe), EXACT[name](*probe)
    assert abs(pw.psi(f, c, "M1") - ex["psi"][0]) <= TOL
    assert abs(pw.psi(f, c, "M2") - ex["psi"][1]) <= TOL
    b12, b21 = pw.psi_block(f, c, ["M1", "M2"]), pw.psi_block(f, c, ["M2", "M1"])
    assert abs(b12 - ex["block"]) <= TOL and abs(b12 - b21) <= TOL, (b12, b21, ex["block"])


def test_s1_block_is_the_joint_conditional(big):
    """-0.56, not the one-at-a-time sweep's -56/149 = -0.376."""
    pw, f = _pw(big, "S1")
    b = pw.psi_block(f, {"M1": 0.0, "M2": 0.0}, ["M1", "M2"])
    assert abs(b + 0.56) <= TOL and abs(b + 56 / 149) > 0.1


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_identity_with_last_position(big, name):
    """Psi_j + v(all) - v(all without j) = 0 where conditioning equals fixing."""
    pw, f = _pw(big, name)
    for probe in PROBES:
        ex = EXACT[name](*probe)
        lam = ex["lam"]
        assert abs(pw.psi(f, _ref(name, probe), "M1") + lam["12"] - lam["2"]) <= TOL
        assert abs(pw.psi(f, _ref(name, probe), "M2") + lam["12"] - lam["1"]) <= TOL


def test_identity_fails_on_s3(big):
    pw, f = _pw(big, "S3")
    c, ex = _ref("S3", (0.0, 0.0)), EXACT["S3"](0.0, 0.0)
    psi1, alt2 = pw.psi(f, c, "M1"), pw.psi_alt(f, c, "M2")
    assert abs(-psi1 + 0.128) <= TOL and abs((ex["lam"]["12"] - ex["lam"]["2"]) + 0.45) <= TOL   # -0.128 vs -0.45
    assert abs(-alt2 + 0.192) <= TOL and abs(ex["lam"]["2"] - 0.13) <= TOL                      # -0.192 vs 0.13


def test_s4_alternatives(big):
    _, X, _ = big["S1"]
    pw = PWShapRegression(X, "T", ["M1", "M2"], seed=1)
    f4 = lambda D: 0.7 * D["M1"].to_numpy(float)
    c, ex = {"M1": 0.0, "M2": 0.0}, cf.s4()
    assert abs(pw.psi(f4, c, "M1") - ex["psi"][0]) <= TOL                                         # -0.376
    assert abs(-pw.psi_alt(f4, c, "M1") - 0.56) <= TOL and abs(-pw.psi_alt(f4, c, "M2") - 0.184) <= TOL
    assert abs(pw.psi_alt(f4, c, "M1") - ex["alt"][0]) <= TOL and abs(pw.psi_alt(f4, c, "M2") - ex["alt"][1]) <= TOL


@pytest.mark.parametrize("variant", [False, True])
def test_two_chain_blocks(variant):
    """Frequency-table branch: block effects of each path at all 16 configurations against enumeration.
    N = 2,000,000 and J = 200,000: the rarest pool (treated, B1 = 0, B2 = 1) then has about 30,000 rows;
    at N = 400,000 it has 5,900 and the variant misses one cell by 0.014 from sampling alone."""
    scm = TwoChains(variant)
    X = scm.sample(2_000_000, seed=3)
    pw = PWShapRegression(X, "T", scm.mediators, J=200_000, seed=1)
    f = lambda D: scm.f_true(D)
    fv = (lambda T, a1, a2, b1, b2: 0.3 * T + (1 + T) * a2 * b2) if variant else (lambda T, a1, a2, b1, b2: 0.3 * T + a2 * b2)
    for cfg in itertools.product((0, 1), repeat=4):
        c = dict(zip(scm.mediators, map(float, cfg)))
        d_all = fv(1, *cfg) - fv(0, *cfg)
        for block, keep in ((["A1", "A2"], ("B1", "B2")), (["B1", "B2"], ("A1", "A2"))):
            pins = {k: cfg[scm.mediators.index(k)] for k in keep}
            exact = d_all - (tce.mean(fv, 1, pins) - tce.mean(fv, 0, pins))
            assert abs(pw.psi_block(f, c, block) - exact) <= TOL, (cfg, block)


def test_alternative_draws_free_w_from_the_arm(big):
    """S3: T = 1 never occurs with W > 0.8, so f is 0 on the rows of both arms and psi_alt is 0.
    A Gaussian draw of W given the arm would reach W > 0.8 and give about -0.009."""
    pw, _ = _pw(big, "S3")
    f = lambda D: D["T"].to_numpy(float) * (D["W"].to_numpy(float) > 0.8) * (D["M1"].to_numpy(float) > 0)
    assert pw.psi_alt(f, {"M1": 0.0, "M2": 0.0, "W": 0.3}, "M1") == 0

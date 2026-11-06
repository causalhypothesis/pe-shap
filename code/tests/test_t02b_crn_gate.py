"""T2b: as T2 at J = 100,000 with five repeats; a cell is flagged when
|mean - exact| > max(5 SE, 0.002), the floor 0.001 for S2's ATE (0.01).

Each repeat draws its own data set (N = 400,000) as well as its own CRN seed, so the SE covers the
error of the fitted mechanisms too.  With one fixed data set and five CRN seeds the SE covers the
Monte Carlo error only and the gate flags ordinary fitting error (on S1 the fitted slope of M2 on M1
is 0.7018 on seed 12345, which moves mu(1, {}) to 1.062 against 1.06)."""
import numpy as np, pytest
from pe_shap.scm import S1, S2, S3
from pe_shap.sampler import MechanismSampler, make_mean_do
from pe_shap.game import coalition_table, shapley_exact
from tests.conftest import PROBES, N_BIG
from tests.test_t02_sampler_a import EXACT, COALS, PLAYERS

SCMS = {"S1": S1(), "S2": S2(), "S3": S3("submitted"), "S3c": S3("clipped")}


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S3c"])
def test_crn_gate(name):
    scm = SCMS[name]
    fits = [MechanismSampler(scm.sample(N_BIG, seed=1000 + r), scm.graph, "T", scm.pretreatment, scm.mediators,
                             seed=r).fit() for r in range(5)]
    flagged = []
    for probe in PROBES:
        ex = EXACT[name](*probe)
        cells = {}
        for A in fits:
            tab = coalition_table(make_mean_do(A, scm.f_true, 100_000), PLAYERS, {"M1": probe[0], "M2": probe[1]})
            phi = shapley_exact(tab["v"], tab["players"])
            for S, key in COALS:
                cells.setdefault(("mu_t", key), []).append(tab["mu"][S][0])
                cells.setdefault(("mu_tprime", key), []).append(tab["mu"][S][1])
                if S:
                    cells.setdefault(("v", key), []).append(tab["v"][S])
            cells.setdefault(("ate", ""), []).append(tab["ate"])
            cells.setdefault(("phi", "M1"), []).append(phi["M1"]); cells.setdefault(("phi", "M2"), []).append(phi["M2"])
        target = {("ate", ""): ex["ate"], ("phi", "M1"): ex["phi"][0], ("phi", "M2"): ex["phi"][1]}
        for _, key in COALS:
            target[("mu_t", key)] = ex["mu"][key]["t"]; target[("mu_tprime", key)] = ex["mu"][key]["tprime"]
            if key != "0":
                target[("v", key)] = ex["lam"][key]
        for cell, xs in cells.items():
            xs = np.asarray(xs); se = xs.std(ddof=1) / np.sqrt(len(xs))
            floor = 0.001 if (name == "S2" and cell == ("ate", "")) else 0.002
            if abs(xs.mean() - target[cell]) > max(5 * se, floor):
                flagged.append((probe, cell, round(xs.mean(), 4), round(target[cell], 4), round(se, 5)))
    assert not flagged, flagged

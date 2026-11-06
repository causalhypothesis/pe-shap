"""Data loaders with assertions, splits, and mediator-dependence diagnostics.
Test T7 runs the assertions.
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

COMPAS_FEATURES = ["race", "age", "priors_count", "juv_fel_count", "felony"]
COMPAS_MEDIATORS = ["priors_count", "juv_fel_count", "felony"]
COMPAS_REFERENCE = {"age": 31, "priors_count": 2, "juv_fel_count": 0, "felony": 1}
GERMAN_FEATURES = ["female", "age", "savings", "amount"]
GERMAN_MEDIATORS = ["savings", "amount"]
GERMAN_REFERENCE = {"age": 34, "savings": 1, "amount": 1569}
SAVINGS_CODES = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3, "missing": 4}


def load_compas(path) -> pd.DataFrame:
    """ProPublica two-year file with the standard cohort filter; label two_year_recid.

    The filter is ProPublica's (screening within 30 days of arrest; is_recid != -1;
    charge degree not 'O'; score_text present), then the two race groups.  Rows with a
    missing score_text are dropped explicitly (not coerced) and their number is printed.
    """
    d = pd.read_csv(path)
    n0 = len(d)
    d = d[(d.days_b_screening_arrest >= -30) & (d.days_b_screening_arrest <= 30)]
    d = d[d.is_recid != -1]
    d = d[d.c_charge_degree != "O"]
    n_missing_score = int(d.score_text.isna().sum() + (d.score_text == "N/A").sum())
    d = d[d.score_text.notna() & (d.score_text != "N/A")]
    print(f"load_compas: {n0} rows read; {n_missing_score} dropped for a missing score_text")
    d = d[d.race.isin(["African-American", "Caucasian"])].copy()
    out = pd.DataFrame({
        "id": d["id"].to_numpy(),
        "race": (d.race == "African-American").astype(int).to_numpy(),
        "age": d.age.to_numpy(int),
        "priors_count": d.priors_count.to_numpy(int),
        "juv_fel_count": d.juv_fel_count.to_numpy(int),
        "felony": (d.c_charge_degree == "F").astype(int).to_numpy(),
        "y": d.two_year_recid.to_numpy(int)})
    assert len(out) == 5278, len(out)
    assert out.race.sum() == 3175 and (1 - out.race).sum() == 2103
    assert out.id.is_unique
    assert abs(out.y.mean() - 0.470) < 0.0015, out.y.mean()
    assert out.age.median() == 31 and out.priors_count.median() == 2 and out.juv_fel_count.median() == 0
    assert abs(out.felony.mean() - 0.652) < 0.0015, out.felony.mean()
    return out.reset_index(drop=True)


def load_german(path) -> pd.DataFrame:
    """Ten-column german_credit.csv as shipped; explicit encodings (female=1, bad=1, savings codes)."""
    d = pd.read_csv(path, index_col=0)
    sav = d["Saving accounts"].fillna("missing").map(SAVINGS_CODES)
    assert sav.notna().all(), "unexpected savings label"
    out = pd.DataFrame({
        "female": (d["Sex"].str.lower() == "female").astype(int).to_numpy(),
        "age": d["Age"].to_numpy(int),
        "savings": sav.to_numpy(int),
        "amount": d["Credit amount"].to_numpy(int),
        "y": (d["Risk"].str.lower() == "bad").astype(int).to_numpy()})
    assert len(out) == 1000 and out.female.sum() == 310 and out.y.sum() == 300
    assert (out.savings == 4).sum() == 183
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int, test_size: float, id_col: str | None = None):
    """Stratified split by y with the given seed; asserts person-disjointness when id_col is given."""
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    if id_col is not None:
        assert not set(tr[id_col]) & set(te[id_col]), "train and test share a person"
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def _partial_rank_corr(x, y, age):
    A = np.vander(age.astype(float), 4)          # cubic polynomial of age
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def mediator_dependence(df, arm: str, age: str, mediators: list, ref_age: float, h: float = 0.4) -> list[dict]:
    """EXPLORATORY association of the mediators given the arm and age: Spearman rho and p
    within the age window per arm, and the partial rank correlation after a cubic in age.

    This is not a test of conditional independence given exact age (mediators that are
    independent given age are correlated inside any window of positive width, and a cubic
    rank residual is not an exact adjustment), so it cannot show that the data "reject" a
    graph.  Every statistic is computed WITHIN an arm, never pooled across arms, because
    pooling creates association through the treatment alone.
    """
    sd = df[age].std(ddof=1); rows = []
    for tau in (0, 1):
        D = df[df[arm] == tau]
        Dw = D[(D[age] - ref_age).abs() <= h * sd]
        for i, a in enumerate(mediators):
            for b in mediators[i + 1:]:
                rho, p = spearmanr(Dw[a], Dw[b])
                rows.append({"arm": tau, "pair": (a, b), "h": h, "n_window": len(Dw), "rho_window": float(rho),
                             "p_window": float(p), "partial_rho_given_age": _partial_rank_corr(D[a], D[b], D[age])})
    return rows

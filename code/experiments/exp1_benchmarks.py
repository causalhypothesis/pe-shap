#!/usr/bin/env python3
"""EXP-1, 1b, 1c: Table 1, tab:bench-full and tab:causal-shapley on S1-S3.

Protocol as submitted (compare_methods_on_benchmarks.py): R = 20 runs, seed r = 0..19; N = 10,000 rows
from the SCM (scm.S1-S3 reproduce the shipped generators exactly for the same seed); the shipped split
(a permutation with seed r, the first 500 rows held out, 9,500 train); MLP regressor (64, 64),
max_iter = 1000, random_state = 42, inputs in the shipped order.  References: every untreated held-out
row at its own mediator values (about 250).
  PE-SHAP  MechanismSampler, OLS mechanisms on the training rows, CRN seed r; 2,000 draws per
           row and coalition; the ATE with 200,000 draws once per run.
  oracle   the true mechanisms with the same model, 4,000 draws per row, the same reference rows (EXP-1c).
  PW-SHAP  PWShapRegression, 20,000 draws: Psi_1, Psi_2, the block and the alternatives, in
           PW-SHAP's own sign (the tables reverse it).
  CS       CausalShapleyT on the first 100 untreated held-out rows, 2,000 draws (EXP-1b).
  NIE      model-based, true mechanisms, 200,000 draws.
  S3       the submitted law; "supported" = untreated references with -0.2 < W < 0.8.  PW-SHAP is computed
           on those rows only; PE-SHAP and the oracle are averaged over both sets.
--fast: R = 2 and small budgets, meta.mode = "fast" (test T8).  --print: the table bodies from the result
file, rounded once with pe_shap.io, as the paper prints them.  --ideal:
writes results/exp1_ideal.json, the ideal-model (closed-form) values that the tables print in brackets.
Usage (from code/): python experiments/exp1_benchmarks.py [--fast] [--out ../results/exp1.json] [--print | --ideal]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

CODE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np, pandas as pd                                                      # noqa: E402
from pe_shap import scm as scms, estimands as est, closed_forms as cf                 # noqa: E402
from pe_shap.sampler import MechanismSampler, make_mean_do                           # noqa: E402
from pe_shap.baselines.pwshap import PWShapRegression                                # noqa: E402
from pe_shap.baselines.causal_shapley import CausalShapleyT                          # noqa: E402
from pe_shap.models import fit_mlp_regressor, as_callable                            # noqa: E402
from pe_shap.io import write_result, pm, fmt, fmt_exact                            # noqa: E402

FULL = dict(R=20, J=2000, J_ate=200_000, J_oracle=4000, J_pw=20_000, n_cs=100, J_cs=2000, J_nie=200_000)
FAST = dict(R=2, J=200, J_ate=2000, J_oracle=200, J_pw=200, n_cs=3, J_cs=200, J_nie=2000)
N, N_EVAL = 10_000, 500
P = {"M1": ["M1"], "M2": ["M2"]}
SCMS = {"S1": scms.S1(), "S2": scms.S2(), "S3": scms.S3("submitted")}
MODEL_COLS = {"S1": ["T", "M1", "M2"], "S2": ["T", "M1", "M2"], "S3": ["T", "W", "M1", "M2"]}   # shipped order
DEPS = ["experiments/exp1_benchmarks.py", "pe_shap/scm.py", "pe_shap/sampler.py", "pe_shap/game.py",
        "pe_shap/estimands.py", "pe_shap/closed_forms.py", "pe_shap/models.py", "pe_shap/io.py",
        "pe_shap/baselines/pwshap.py", "pe_shap/baselines/causal_shapley.py"]


def shipped_split(D: pd.DataFrame, seed: int):
    """_common.split_train_eval: a permutation with the run's seed; the first N_EVAL rows are held out."""
    D = D.iloc[np.random.default_rng(seed).permutation(len(D))].reset_index(drop=True)
    return D.iloc[N_EVAL:].reset_index(drop=True), D.iloc[:N_EVAL].reset_index(drop=True)


def _avg(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def one_run(name: str, r: int, b: dict) -> dict:
    scm = SCMS[name]
    train, held = shipped_split(scm.sample(N, seed=r), r)
    cols = MODEL_COLS[name]
    f = as_callable(fit_mlp_regressor(train[cols].to_numpy(float), train["Y"].to_numpy(float), seed=42), cols)
    Xtr = train[scm.pretreatment + ["T"] + scm.mediators]
    refs = held[held["T"] == 0].reset_index(drop=True)
    supported = refs["W"].between(-0.2, 0.8, inclusive="neither").to_numpy() if name == "S3" else np.ones(len(refs), bool)

    A = MechanismSampler(Xtr, scm.graph, "T", scm.pretreatment, scm.mediators, seed=r).fit()
    md = make_mean_do(A, f, b["J_ate"]); ate = md(1, {}) - md(0, {})
    O = scms.OracleSampler(scm, Xtr, J=b["J_oracle"], seed=r)
    ate_o = O.mean_do(f, 1, {}, J=b["J_ate"]) - O.mean_do(f, 0, {}, J=b["J_ate"])
    pw = PWShapRegression(Xtr, "T", scm.pretreatment + scm.mediators, pretreatment=scm.pretreatment, J=b["J_pw"], seed=r)

    per, per_pw = [], []
    for i, row in refs.iterrows():
        ref = {"M1": row["M1"], "M2": row["M2"]}
        g, go = est.pe_game(A, f, P, ref, J=b["J"], ate=ate), est.pe_game(O, f, P, ref, J=b["J_oracle"], ate=ate_o)
        per.append({"lam1": g["v"][("M1",)], "lam2": g["v"][("M2",)], "lam12": g["v"][("M1", "M2")],
                    "phi1": g["phi"]["M1"], "phi2": g["phi"]["M2"],
                    "o_phi1": go["phi"]["M1"], "o_phi2": go["phi"]["M2"], "o_lam12": go["v"][("M1", "M2")]})
        if supported[i]:
            c = {**ref, **({"W": row["W"]} if name == "S3" else {})}
            per_pw.append({"psi1": pw.psi(f, c, "M1"), "psi2": pw.psi(f, c, "M2"), "block": pw.psi_block(f, c, ["M1", "M2"]),
                           "alt1": pw.psi_alt(f, c, "M1"), "alt2": pw.psi_alt(f, c, "M2")})

    cs = CausalShapleyT(A, "T", scm.pretreatment, J=b["J_cs"], seed=r)
    players = scm.pretreatment + ["T", "M1", "M2"]
    cs_rows = [cs.phi_T(f, {**{k: row[k] for k in scm.pretreatment}, "T": 0, "M1": row["M1"], "M2": row["M2"]}, players)
               for _, row in refs.head(b["n_cs"]).iterrows()]

    m = _avg(per, per[0]); mp = _avg(per_pw, per_pw[0])
    out = {"ate": ate, "lam": {"1": m["lam1"], "2": m["lam2"], "12": m["lam12"]}, "phi": {"M1": m["phi1"], "M2": m["phi2"]},
           "pw": {"psi1": mp["psi1"], "psi2": mp["psi2"], "block": mp["block"]}, "pw_alt": {"1": mp["alt1"], "2": mp["alt2"]},
           "nie": est.nie_model_based(scm, f, J=b["J_nie"], seed=r),
           "oracle": {"phi": {"M1": m["o_phi1"], "M2": m["o_phi2"]}, "lam12": m["o_lam12"]},
           "cs": {"dir": float(np.mean([x["direct"] for x in cs_rows])), "ind": float(np.mean([x["indirect"] for x in cs_rows]))},
           "n_ref": len(refs)}
    if name == "S3":
        s = _avg([p for p, ok in zip(per, supported) if ok], per[0])
        out["supported"] = {"phi": {"M1": s["phi1"], "M2": s["phi2"]}, "lam12": s["lam12"], "n_ref": int(supported.sum()),
                            "oracle": {"phi": {"M1": s["o_phi1"], "M2": s["o_phi2"]}, "lam12": s["o_lam12"]}}
    return out


# ------------------------------------------------------------------------------------------- tables
def exact_population() -> dict:
    """Untreated-population values of the true function (closed forms, 3-point Gauss-Hermite per dimension,
    exact for these polynomials of degree <= 2 in the pins), and Causal Shapley's untreated means."""
    x, w = np.polynomial.hermite_e.hermegauss(3); w = w / w.sum()
    L = {"S1": [[0.5, 0], [0.35, 0.5]], "S2": [[0.5, 0], [0, 0.5]], "S3": [[0.5, 0], [0.35, 0.5]]}
    fn = {"S1": lambda a, c: cf.s1(a, c), "S2": cf.s2, "S3": lambda a, c: cf.s3(a, c)}
    out = {}
    for s in L:
        acc = {}
        for i in range(3):
            for j in range(3):
                m1, m2 = np.array(L[s]) @ np.array([x[i], x[j]])
                g = fn[s](float(m1), float(m2))
                for k, v in (("phi1", g["phi"][0]), ("phi2", g["phi"][1]), ("lam1", g["lam"]["1"]), ("lam2", g["lam"]["2"]),
                             ("lam12", g["lam"]["12"]), ("psi1", g["psi"][0]), ("psi2", g["psi"][1]), ("block", g["block"]),
                             ("alt1", g["alt"][0]), ("alt2", g["alt"][1])):
                    acc[k] = acc.get(k, 0.0) + w[i] * w[j] * v
        acc["nie"] = fn[s](0.0, 0.0)["nie"]
        out[s] = acc
    out["S1"].update(cs_dir=cf.s1()["cs_dir"][0], cs_ind=cf.s1()["cs_ind"][0])
    out["S2"].update(cs_dir=cf.s2()["cs_dir"][0], cs_ind=cf.s2()["cs_ind"][0])
    m3 = cf.s3_cs_means("submitted")["untreated"]
    out["S3"].update(cs_dir=m3["dir"], cs_ind=m3["ind"])
    return out


def write_ideal(results_path, out_path):
    """The ideal-model values (closed forms) next to the result file, for the brackets in the tables and the
    number checker; 'derived' holds numbers the text quotes that combine both files."""
    E = exact_population()
    S = json.loads(pathlib.Path(results_path).read_text())["summary"]
    gap = max(abs(S[f"S3.supported.{k}"]["mean"] - E["S3"][e]) for k, e in (("phi.M1", "phi1"), ("phi.M2", "phi2"), ("lam12", "lam12")))
    doc = {**{s: {k: float(v) for k, v in E[s].items()} for s in SCMS}, "derived": {"s3_gap_max": float(gap)}}
    pathlib.Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print("wrote", out_path)


def print_tables(path):
    r"""Table bodies in the paper's layout: estimate {\pm SD} and the ideal model's value in brackets."""
    S = json.loads(pathlib.Path(path).read_text())["summary"]
    E = exact_population()
    def cell(key, ideal, neg=False):
        m = -S[key]["mean"] if neg else S[key]["mean"]
        return f"${pm(m, S[key]['sd'], 2, small=True)}\\,{{\\scriptstyle({fmt_exact(-ideal if neg else ideal)})}}$"
    row = lambda label, cells: label + " & " + " & ".join(c if c else "" for c in cells) + r" \\"
    pe = lambda s, sup="": [cell(f"{s}.{sup}phi.M1", E[s]["phi1"]), cell(f"{s}.{sup}phi.M2", E[s]["phi2"]),
                            cell(f"{s}.{sup}lam12" if sup else f"{s}.lam.12", E[s]["lam12"])]
    pw = lambda s: [cell(f"{s}.pw.psi1", E[s]["psi1"], True), cell(f"{s}.pw.psi2", E[s]["psi2"], True), cell(f"{s}.pw.block", E[s]["block"], True)]
    print("% Table 1 (tab:main-results): S3 PE-SHAP on the supported references; PW-SHAP sign reversed")
    print(row("PE", pe("S1") + pe("S2") + pe("S3", "supported.")))
    print(row("PW", pw("S1") + pw("S2") + pw("S3")))
    print(row(r"$\mathrm{NIE}_{\hat Y}$", ["---", "---", cell("S1.nie", E["S1"]["nie"]), "---", "---", cell("S2.nie", E["S2"]["nie"]),
                                           "---", "---", cell("S3.nie", E["S3"]["nie"])]))
    print("\n% tab:bench-full")
    orc = lambda s, sup="": [cell(f"{s}.{sup}oracle.phi.M1", E[s]["phi1"]), cell(f"{s}.{sup}oracle.phi.M2", E[s]["phi2"]),
                             cell(f"{s}.{sup}oracle.lam12", E[s]["lam12"])]
    print(row("PE, true mediator distributions (oracle)", orc("S1") + orc("S2") + orc("S3", "supported.")))
    lam = lambda s: [cell(f"{s}.lam.1", E[s]["lam1"]), cell(f"{s}.lam.2", E[s]["lam2"]), ""]
    print(row(r"$\lambda_{M_1}$, $\lambda_{M_2}$ (raw)", lam("S1") + lam("S2") + lam("S3")))
    alt = lambda s: [cell(f"{s}.pw_alt.1", E[s]["alt1"], True), cell(f"{s}.pw_alt.2", E[s]["alt2"], True), ""]
    print(row(r"PW, alternative ($-\Psi^{\mathrm{alt}}_j$)", alt("S1") + alt("S2") + alt("S3")))
    print("\n% tab:causal-shapley")
    for s in SCMS:
        cs = lambda k: (f"${'+' if S[f'{s}.cs.{k}']['mean'] > 0 else ''}{pm(S[f'{s}.cs.{k}']['mean'], S[f'{s}.cs.{k}']['sd'], 3)}"
                        f"\\,({fmt(E[s]['cs_' + k], 3, sign=True)})$")
        print(f"{s} & {cs('dir')} & {cs('ind')}\\\\")
    print(f"\n% n_ref: mean {np.mean([S[f'{s}.n_ref']['mean'] for s in SCMS]):.0f}; S3 supported {S['S3.supported.n_ref']['mean']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "exp1.json"))
    ap.add_argument("--print", action="store_true", help="print the table bodies from --out and stop")
    ap.add_argument("--ideal", action="store_true", help="write the ideal-model values next to --out and stop")
    a = ap.parse_args()
    if a.print:
        return print_tables(a.out)
    if a.ideal:
        return write_ideal(a.out, str(pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + "_ideal.json")))
    b = FAST if a.fast else FULL
    runs = []
    for r in range(b["R"]):
        rec = {"seed": r}
        for name in SCMS:
            t0 = time.time()
            rec[name] = one_run(name, r, b)
            print(f"run {r} {name}: phi ({rec[name]['phi']['M1']:+.3f}, {rec[name]['phi']['M2']:+.3f}) "
                  f"lam12 {rec[name]['lam']['12']:+.3f} | {time.time() - t0:.0f}s", flush=True)
        runs.append(rec)
    write_result(a.out, "experiments/exp1_benchmarks.py", {**b, "N": N, "n_eval": N_EVAL, "seeds": list(range(b["R"])),
                 "s3_law": "submitted", "supported": "-0.2 < W < 0.8", "model": "MLPRegressor(64, 64), max_iter=1000, random_state=42"},
                 runs, deps=DEPS, mode="fast" if a.fast else "full")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

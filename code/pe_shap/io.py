"""Result files, summaries and number formatting.

The paper prints the values of the result files with the formatting below.
"""
from __future__ import annotations
import datetime, json, pathlib, platform, subprocess, sys
import numpy as np


def _versions():
    import numpy, pandas, sklearn, scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def _git_hash():
    """HEAD if the tree is a repository, else "no-git" (a downloaded archive); never raises."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


SOURCE_SET = ("pe_shap", "experiments", "configs")   # the tested source and configuration set
PRODUCTION = pathlib.Path(__file__).resolve().parents[2] / "results"   # the saved production results


def source_digest(root: str = ".") -> str:
    """sha256 over the sorted relative paths and contents of the SOURCE_SET under root (*.py and
    *.json): the identity of the code.
    Results, dates, runtimes and platform strings are not part of it."""
    import hashlib
    h = hashlib.sha256()
    base = pathlib.Path(root)
    for sub in SOURCE_SET:
        for f in sorted(list((base / sub).rglob("*.py")) + list((base / sub).rglob("*.json"))):
            h.update(str(f.relative_to(base)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def deps_digest(files, root: str = ".") -> str:
    """sha256 over the sorted relative paths and contents of the files a campaign depends on (its
    experiment script, the pe_shap modules it imports, its config files).  Recorded in the result
    file, so a change to an UNRELATED file never flags a campaign as stale, and a change to a
    dependency does."""
    import hashlib
    h = hashlib.sha256()
    base = pathlib.Path(root)
    for f in sorted(files):
        h.update(f.encode()); h.update((base / f).read_bytes())
    return h.hexdigest()


def _nullify(o, nullable, prefix=""):
    """Replace non-finite floats by None where the key path starts with a nullable prefix.
    Any other non-finite value is left in place and makes json.dumps(allow_nan=False) raise."""
    if isinstance(o, dict):
        return {k: _nullify(v, nullable, f"{prefix}.{k}" if prefix else k) for k, v in o.items()}
    if isinstance(o, list):
        return [_nullify(v, nullable, prefix) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o) and any(prefix.startswith(n) for n in nullable):
        return None
    return float(o) if isinstance(o, (np.floating, np.integer)) else o


def write_result(path, script: str, config: dict, runs: list, extra: dict | None = None,
                 deps: list | None = None, mode: str = "full", nullable: tuple = ()):
    """JSON with meta (script, date, git hash if any, source and dependency digests, mode), config
    (incl. seeds), runs, summary.

    nullable: prefixes of leaves that MAY be undefined in a run (a donor contrast with no donors,
    a stratum with no rows).  Such leaves are written as JSON null together with the reason and
    counts the script records beside them; summarise() treats them as optional.  A script writes
    EVERY leaf of its schema in EVERY run (None when undefined) and never omits a key, so that an
    all-undefined leaf still appears in the summary with n = 0 and n_missing = R.  Every other leaf
    must be finite: the file is serialised with allow_nan=False, so an undeclared NaN raises here
    instead of reaching a table.  mode is "full" or "fast".
    A --fast run must never overwrite a production result, so it needs an --out elsewhere.
    """
    if mode != "full" and pathlib.Path(path).resolve().parent == PRODUCTION:
        raise ValueError(f"--fast must not write to {PRODUCTION}: use --out elsewhere, e.g. --out ../results_fast/{pathlib.Path(path).name}")
    doc = {"meta": {"script": script, "date": datetime.date.today().isoformat(), "git": _git_hash(),
                    "source_digest": source_digest(), "deps_digest": deps_digest(deps) if deps else None,
                    "deps": sorted(deps) if deps else [], "mode": mode, "nullable": list(nullable),
                    "versions": _versions()},
           "config": config, "runs": runs, "summary": summarise(runs, optional=nullable)}
    if extra:
        doc.update(extra)
    doc = _nullify(doc, tuple(f"runs.{n}" for n in nullable) + tuple(f"summary.{n}" for n in nullable))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return doc


def _flatten(d, prefix=""):
    """Numeric leaves of a nested dict as (dotted key, float); None (an undefined leaf) is
    yielded as NaN so that strict mode can count it as missing."""
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _flatten(v, key)
        elif v is None:
            yield key, float("nan")
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            yield key, float(v)


def summarise(runs: list, optional: tuple = (), strict: bool = True) -> dict:
    """mean, sd (ddof=1), n, positives, negatives for every numeric leaf of the runs.

    strict (default): every leaf must be present in every run and finite, except leaves
    whose key starts with a prefix in `optional` (declared-undefined quantities such as a
    donor contrast with no donors; those are summarised over the runs that have them and
    carry n_missing).  A missing or non-finite leaf elsewhere raises, so that a table
    cannot be written from an incomplete or NaN result.
    """
    cols, R = {}, len(runs)
    for r in runs:
        for k, v in _flatten(r):
            cols.setdefault(k, []).append(v)
    out = {}
    for k, xs in cols.items():
        a = np.asarray(xs, float)
        is_opt = any(k.startswith(pfx) for pfx in optional)
        if strict and not is_opt and (len(a) != R or not np.all(np.isfinite(a))):
            raise ValueError(f"summarise: leaf {k!r} present in {len(a)}/{R} runs, finite in {int(np.isfinite(a).sum())}")
        a = a[np.isfinite(a)]
        if len(a) == 0:
            out[k] = {"mean": float("nan"), "sd": float("nan"), "n": 0, "n_missing": R, "pos": 0, "neg": 0,
                      "min": float("nan"), "max": float("nan")}
            continue
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a)),
                  "n_missing": int(R - len(a)),
                  "pos": int((a > 0).sum()), "neg": int((a < 0).sum()), "min": float(a.min()), "max": float(a.max())}
    return out


def fmt(x: float, d: int = 2, sign: bool = False) -> str:
    """Round once from the raw value; a value that rounds to zero prints without a sign."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0.0:
        s = f"{0:.{d}f}"
    return s


def pm(mean: float, sd: float, d: int = 2, small: bool = False) -> str:
    """'0.28\\pm0.02' or Table-1 style '.28{\\scriptstyle\\pm.02}' (small=True drops the leading zero)."""
    m, s = fmt(mean, d), fmt(sd, d)
    if small:
        m, s = m.replace("0.", ".", 1) if abs(mean) < 1 else m, s.replace("0.", ".", 1)
        return f"{m}{{\\scriptstyle\\pm{s}}}"
    return f"{m}\\pm{s}"


def fmt_exact(x: float) -> str:
    """An ideal-model (closed-form) value as Table 1 prints it in brackets: three decimals when the
    third is not zero, else two; no leading zero ('.065', '-.32', '.00')."""
    s = fmt(x, 3) if abs(round(x, 3) - round(x, 2)) > 1e-9 else fmt(x, 2)
    return s.replace("0.", ".", 1)


def triple(a: float, b: float, d: int = 2):
    """Formatted (a, b, a+b) that add up as printed; d, d+1 or d+2 decimals.

    Raises if no choice up to d+2 decimals adds up exactly (the caller then prints the
    sum with the words "rounded" or changes the layout); it never returns a non-adding
    triple silently.
    """
    for dd in (d, d + 1, d + 2):
        fa, fb, fs = fmt(a, dd), fmt(b, dd), fmt(a + b, dd)
        if abs(float(fa) + float(fb) - float(fs)) < 1e-12:
            return fa, fb, fs
    raise ValueError(f"triple: {a} + {b} does not add up as printed at {d}..{d+2} decimals")


def table_body(rows: list[list[str]]) -> str:
    """A tabular BODY for pasting into paper.tex (rows joined by ' & ' and '\\\\')."""
    return "\n".join(" & ".join(r) + r"\\" for r in rows)

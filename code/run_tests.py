#!/usr/bin/env python3
"""Run the tests (T1-T8, T10-T12) with pytest.

T7 needs the data files of data/README.md; when they are absent it is left out, with a message.
Usage (from code/): python run_tests.py"""
import pathlib, subprocess, sys

data_absent = not all(pathlib.Path("data", f).exists() for f in ("compas-scores-two-years.csv", "german_credit.csv"))
ignore = ["--ignore=tests/test_t07_data.py"] if data_absent else []
if data_absent:
    print("data files absent: T7 is left out (see data/README.md)")
sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", *ignore]).returncode)

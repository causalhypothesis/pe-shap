# PE-SHAP

Code for *PE-SHAP: Causally Interpretable Path-Wise Shapley Explanations* (NeurIPS 2026),
by Maksym Buleshnyi, Joshua R. Loftus and Sakina Hansen.

PE-SHAP is the Shapley value of a portion-eliminated game in which every coalition of mediators is
valued by intervention. Its per-mediator values sum, at every reference sample, to the portion of
the effect eliminated by fixing all mediators.

## Install

Python 3.12. From the repository root:

    cd code
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

All commands below are run from `code/`.

## Data

The synthetic experiments need no data. The two real datasets (COMPAS and German Credit) are not
included. `data/README.md` says where to download them and where to put them.

## Tests

    python run_tests.py

Without the data files, the data test (T7) is left out.

## Experiments

`results/` holds the result files behind the paper. Each script below writes one of them.

| Script | In the paper | Result file |
|---|---|---|
| `experiments/exp1_benchmarks.py` | synthetic benchmarks S1-S3 | `results/exp1.json` |
| `experiments/exp2_local.py` | local decompositions on S2 | `results/exp2.json` |
| `experiments/exp3_structures.py` | mediation structures | `results/exp3.json` |
| `experiments/exp4_scaling.py` | larger mediator sets (K = 10, 20) | `results/exp4.json` |
| `experiments/exp5_compas.py` | COMPAS | `results/exp5.json` |
| `experiments/exp6_graphs.py` | COMPAS, other graphs and the age adjustment | `results/exp6.json` |
| `experiments/exp7_faithfulness.py` | COMPAS, faithfulness | `results/exp7.json` |
| `experiments/exp8_german.py` | German Credit | `results/exp8.json` |
| `experiments/exp_two_chains.py` | two interacting chains | `results/exp_two_chains.json` |

Run a script with, for example,

    python experiments/exp1_benchmarks.py

This overwrites the saved result file. `exp6_graphs.py` and `exp7_faithfulness.py` read
`results/exp5.json`, so run `exp5_compas.py` first. The full set takes a few CPU-hours.

For a quick check, `--fast` runs a small version. It must write outside `results/`:

    python experiments/exp1_benchmarks.py --fast --out ../results_fast/exp1.json

To reprint the table cells from the saved results:

    python experiments/exp1_benchmarks.py --print
    python experiments/exp2_local.py --print

The other experiments' numbers are in the `summary` block of their result files. `--ideal`
(exp1, exp2, exp_two_chains) writes the closed-form values that the tables show in brackets
(`results/*_ideal.json`).

## Citation

    @inproceedings{buleshnyi2026peshap,
      title     = {{PE-SHAP}: Causally Interpretable Path-Wise {Shapley} Explanations},
      author    = {Buleshnyi, Maksym and Loftus, Joshua R. and Hansen, Sakina},
      booktitle = {Advances in Neural Information Processing Systems},
      year      = {2026}
    }

## Licence

MIT, see `LICENSE`.

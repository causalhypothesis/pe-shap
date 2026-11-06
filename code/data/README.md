# Data

The two CSV files are not included in this repository. Download them into this folder (`code/data/`).
The loaders in `pe_shap/data.py` check the counts after loading (5,278 defendants; 1,000 rows,
310 women, 300 bad risks, 183 missing savings) and stop if a file differs.

## compas-scores-two-years.csv

- Source: ProPublica, https://github.com/propublica/compas-analysis, commit
  c062fbb5fa84f6c99d82f432c46b405f3a0ca5ec (2016-05-22, the last commit touching the file).
  The repository states no licence.
- Download (from `code/data/`):

      curl -sSLo compas-scores-two-years.csv https://raw.githubusercontent.com/propublica/compas-analysis/c062fbb5fa84f6c99d82f432c46b405f3a0ca5ec/compas-scores-two-years.csv

- 7,214 rows. After the standard filter (`pe_shap.data.load_compas`), 5,278 defendants, one row each.

## german_credit.csv

- Source: the UCI Statlog German Credit data (CC BY 4.0, DOI 10.24432/C5NC77), in the common
  ten-column redistribution, for example https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk.
- Header: an unnamed index column, then Age, Sex, Job, Housing, Saving accounts, Checking account,
  Credit amount, Duration, Purpose, Risk. Save it as `german_credit.csv`.
- 1,000 rows. 310 women, 300 bad risks, 183 missing savings.

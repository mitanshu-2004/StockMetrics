# StockMetrics

Do company fundamentals explain annual stock returns for five Indian IT-services
firms (TCS, Infosys, HCL Technologies, Wipro, Tech Mahindra) over roughly 17 years?
The project answers that two ways and reports both, plus a positive control that
checks the pipeline can find a real effect when one exists.

For each firm it fits a multivariate OLS of annual stock return on five fundamental
predictors: sales growth, EBITDA growth, EBITDA-margin change, PAT growth, and
PAT-margin change. It reports each coefficient's significance, the in-sample fit, a
leave-one-out cross-validated error, and a permutation p-value on the joint R-squared.

## Two questions, kept separate

The same-year regression and the forecasting regression are different claims, so the
code runs and labels them separately.

1. Contemporaneous (`lag=0`): year-t return regressed on year-t fundamental growth.
   This asks whether returns and fundamentals move together within the same year.
2. Predictive (`lag=1`): year-t return regressed on the prior year's fundamental
   growth. This uses only information available before the return period, which is
   what a forecasting claim actually requires.

An earlier version of this project ran only the same-year regression but described
the finding as "fundamentals do not predict returns." That was an overstatement: a
same-year fit says nothing about prediction. The lagged regression now tests the
predictive claim directly instead of borrowing the word.

## Result

After reporting multivariate p-values and applying Benjamini-Hochberg correction
across all 25 coefficient tests, no predictor is significant for any firm, in either
the contemporaneous or the predictive regression.

- Contemporaneous: the strongest raw signal is Infosys EBITDA growth (raw p = 0.013),
  which does not survive correction (FDR p = 0.21). Infosys is the one firm whose
  joint fit beats chance on the permutation test (p = 0.042, R-squared = 0.65), but
  no individual driver in it holds up.
- Predictive: nothing survives FDR here either. TCS sits at the edge on the joint
  permutation test (p = 0.050), and no coefficient is robust.
- Across both regressions, leave-one-out RMSE is well above the in-sample error for
  every firm, so the in-sample R-squared (0.22 to 0.66) reflects overfitting on 17 to
  18 points and 5 predictors, not out-of-sample skill.

Two borderline joint fits across ten firm-by-horizon tests is what you expect by
chance at alpha = 0.05, and neither replicates across firms. Per-coefficient output
with raw and FDR-adjusted p-values is written to `output/coefficient_tests.csv` and
`output/coefficient_tests_predictive.csv`.

The honest reading at this sample size: for these five firms, in this dataset, annual
fundamental growth shows no robust same-year or one-year-ahead relationship with
annual returns. This is a statement about a small, single-sector sample (5 firms, ~17
years), not a general law about fundamentals.

## Positive control: the pipeline does find real signal

A null result everywhere would be suspect, so the same machinery runs on a
relationship that genuinely exists: how the five sector peers' annual returns co-move.
Here the signal is clear. Every one of the 10 pairs correlates at r between 0.84 and
0.94, and all 10 survive FDR correction (mean |r| = 0.91, strongest HCL vs Tech
Mahindra at r = 0.94). Written to `output/stock_comovement.csv`.

That contrast is the point. The fundamentals-to-returns relationship is undetectable
in this sample while the sector co-movement is strong, and the same FDR-controlled
pipeline reports each correctly. That is evidence the method discriminates signal from
noise rather than always returning null.

## Mapping note

Stock symbols and fundamentals are joined by ISIN, the unambiguous security
identifier, rather than by a hand-maintained code-to-name table. An earlier version
hard-coded that table and mispaired three of the five firms, which produced a spurious
"significant" result. `derive_code_to_name` now builds the mapping from the file and
fails loudly if it ever disagrees with the expected pairing in `constants.py`.

## Run

```bash
pip install -r requirements.txt
python run_analysis.py
```

Reads `data.xls` (committed, a BIFF/OLE2 `.xls` read via `xlrd`) and writes the
regression tables and chart PNGs to `output/`.

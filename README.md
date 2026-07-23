# Short-Panel Temporal-Precedence Triangulation

Reference code, synthetic data, and fixed validation outputs for the MethodsX workflow **Triangulating Elasticity Dynamics in Short Panels with Classical, Bayesian, and Machine-Learning Precedence Tests**.

The workflow is intended for data sets with many candidate variables, repeated panel units, and only a small number of time points per unit. Examples include sectors, products, regions, organizations, patients, or experimental systems observed repeatedly over time.

> **Interpretation boundary:** this package evaluates association, predictive temporal precedence, and forecasting stability. It does not by itself identify structural causal effects. Report supported relations as *temporal precedence* or *Granger-style predictive precedence* unless an independent identification design justifies stronger language.

## Method in one page

The analysis uses a common data contract and several deliberately different evidence families:

1. **Association screen:** Pearson and Spearman relations, preferably after within-unit or two-way adjustment, with circular-shift surrogate calibration.
2. **Classical precedence:** restricted-versus-unrestricted dynamic panel models, complete lag evidence, correction for searching across lag orders, and Benjamini-Hochberg false-discovery-rate control within declared testing families.
3. **Bayesian comparison:** restricted-versus-unrestricted model evidence reported as `BF10`, accompanied by design-rank and prior-sensitivity checks.
4. **Machine-learning screen:** forward-chaining prediction, restricted-model comparison, and circular-shift shadow predictors; raw Random Forest importance is not treated as a significance test.
5. **Forecast validation:** rolling-origin evaluation against a naive benchmark before future paths are interpreted.
6. **Triangulation:** method agreement, unit/layer replication, placebo performance, and sensitivity to common-shock controls are retained in an auditable evidence table.

## Data contract

The compact R interface expects a long data frame with one row per unit-time pair:

| Field | Meaning |
|---|---|
| `unit` | repeated panel unit identifier |
| `time` | ordered time identifier |
| `y` | response series |
| predictor columns | candidate leading variables |

The original empirical project also uses aligned arrays:

- `Y`: numeric matrix with shape **time x unit**;
- `X`: numeric array with shape **predictor x unit x time**.

Every preprocessing decision must preserve temporal order. Imputation, scaling, feature construction, and model fitting should occur within each rolling-origin training window when forecasts or machine-learning evidence are evaluated.

## Repository structure

```text
methodsx-short-panel-triangulation/
├── README.md
├── LICENSE
├── CITATION.cff
├── DESCRIPTION
├── config/
│   └── defaults.yml
├── R/
│   ├── utils.R
│   ├── correlation_forecast.R
│   ├── classical_panel_precedence.R
│   ├── bayesian_panel_precedence.R
│   ├── rf_panel_precedence.R
│   └── triangulate_evidence.R
├── python/
│   ├── requirements.txt
│   ├── generate_validation_data.py
│   └── run_sensitivity_matrix.py
├── data/
│   ├── synthetic_mixed_panel.csv
│   ├── synthetic_null_panel.csv
│   ├── synthetic_common_driver_panel.csv
│   ├── synthetic_structural_break_panel.csv
│   ├── truth_mixed.csv
│   └── truth_all.csv
├── outputs/
│   ├── demo_evidence.csv
│   ├── common_driver_sensitivity.csv
│   ├── simulation_replications.csv
│   ├── simulation_summary.csv
│   └── validation_manifest.json
├── figures/
│   ├── validation_method_support.png
│   ├── common_driver_sensitivity.png
│   └── simulation_detection_rates.png
├── tests/
│   ├── smoke_test.R
│   └── test_reference_outputs.py
├── docs/
│   ├── METHOD_VALIDATION.md
│   └── ORIGINAL_SCRIPT_MAP.md
└── run_demo.R
```

## Python validation workflow

Python 3.11 or later is recommended. From the repository root:

```bash
python -m pip install -r python/requirements.txt
python python/generate_validation_data.py --out . --seed 20260723
python python/run_sensitivity_matrix.py --out outputs --seed 20260723 --replications 100
python tests/test_reference_outputs.py
```

The first script regenerates the synthetic panels, the method-support evidence table, and the common-driver sensitivity comparison. The second produces the 100-replication smoke-test matrix for `T = 20, 28, 40`. The test script checks the fixed decision contract rather than requiring bitwise-identical floating-point output.

## R method workflow

R 4.3 or later is recommended. Required packages are listed in `DESCRIPTION`; the Random Forest branch additionally uses `ranger`.

From the repository root:

```bash
Rscript tests/smoke_test.R
Rscript run_demo.R
```

`run_demo.R` demonstrates correlation screening, a rolling-origin baseline check, classical and Bayesian precedence tests, the optional Random Forest branch, and evidence classification on the supplied mixed synthetic panel.

For a release or submission archive, also save:

```r
writeLines(capture.output(sessionInfo()), "outputs/sessionInfo.txt")
```

## Synthetic scenarios

- **Mixed panel:** one linear lagged driver, one nonlinear lagged driver, a null predictor, and a common-driver proxy.
- **Null panel:** no planted predictor-to-response temporal-precedence edge.
- **Common-driver panel:** no true `x_common -> y` edge; used to show why unit effects alone can be misleading and why full time effects or lagged leave-one-unit-out cross-sectional averages should be checked as alternative specifications.
- **Structural-break panel:** a visible mid-sample change, included to force sensitivity reporting rather than to claim that break detection is solved.

The supplied scenarios are implementation diagnostics. They are not universal power, size, or performance guarantees. Before confirmatory use, add simulations calibrated to the intended panel length, number of units, autocorrelation, missingness, effect sizes, nonlinear forms, and cross-sectional dependence.

## Evidence and reporting rules

- Justify `max_lag` from the time-series length and the effective observations left after lagging.
- Preserve every lag-specific result. Report the raw minimum p-value, the lag-search-adjusted p-value, and the across-family FDR-adjusted q-value separately.
- Fit restricted and unrestricted models on exactly the same observations and controls.
- Treat full time fixed effects and lagged leave-one-unit-out cross-sectional averages as alternative common-shock controls unless a rank-identified design supports their joint use.
- Report the Bayes-factor orientation (`BF10` means unrestricted over restricted), threshold, approximation or prior, and sensitivity analysis.
- For Random Forest evidence, report the seed, trees, fold construction, restricted and full predictors, forward MSE gain, positive-fold share, number of shadows, and shadow decision rule.
- Preserve disagreement among methods. A one-method finding is provisional, not consensus.
- Validate forecasts with rolling origins and a simple benchmark; report fold losses and interval coverage rather than only the final forecast path.
- Archive seeds, software versions, data checksums, exclusion logs, testing-family definitions, and all long-form evidence tables.

## Relation to the empirical trade application

This compact package contains no proprietary or provider-restricted empirical data. `docs/ORIGINAL_SCRIPT_MAP.md` maps each methodological component to the full `GRAV_PAPER` workflow, including the OLS/PPML gravity engine, panel tests, shadow correction, FDR/placebo diagnostics, and forecasting scripts. Users must obtain empirical data from the original providers and follow their terms.

The intended public locations are:

- compact method package: `https://github.com/kzst/methodsx-short-panel-triangulation`;
- full empirical project after acceptance: `https://github.com/kzst/GRAV_PAPER`.

## License and citation

The reference code is released under the MIT License. Cite the associated MethodsX article and the software release. Update `CITATION.cff` with the final article DOI and archive DOI when available.

# Validation notes

The supplied synthetic data are diagnostics, not universal power benchmarks.

- `synthetic_mixed_panel.csv` contains one linear lagged driver, one nonlinear lagged driver, one null predictor, and one variable correlated through a common driver.
- `synthetic_null_panel.csv` contains no temporal-precedence signal.
- `synthetic_common_driver_panel.csv` demonstrates why unit effects alone can leave common-shock confounding. The reference output compares three specifications: unit effects only; unit and time effects; and unit effects with lagged leave-one-unit-out cross-sectional averages. The latter two are alternatives, not a combined default.
- `synthetic_structural_break_panel.csv` contains a visible coefficient break and is intended for sensitivity checks rather than a claim that break detection is solved.

Reference outputs in `outputs/` were generated with fixed seeds. Re-run the Python scripts to regenerate them, then execute `python tests/test_reference_outputs.py` to check the fixed decision contract. Small numerical differences across BLAS, package, or random-forest implementations are expected. Interpretation should be based on decision stability, not bitwise equality.

The R functions are dependency-light reference implementations. Before confirmatory use, run the supplied smoke test in the target R environment, record `sessionInfo()`, and add domain-calibrated simulations matching the intended panel length, number of units, autocorrelation, missingness, effect size, and cross-sectional dependence.

# Mapping to the GRAV_PAPER source workflow

The MethodsX package is a compact, testable layer over the full research repository. The original scripts remain authoritative for the empirical application.

| Role in the method | Original project script |
|---|---|
| Frequentist panel precedence | `panel_granger_causality.R` |
| Bayesian panel comparison | `bayesian_panel_granger_causality.R` |
| Random Forest panel screen | `rf_panel_granger_causality.R` |
| Shadow-feature correction | `granger_rf_shadow.R` |
| Lag-search correction, FDR, and circular-shift placebo | `panel_granger_robustness_fdr_placebo.R` |
| Compact reporting helpers | `panel_fdr_reporting_helpers.R` |
| Gravity OLS/PPML engine | `calcscripts/gravity_engine.R` |
| OLS-versus-PPML triangulation | `calcscripts/OLS_PPML_COMPARISON.r` |
| Forecast fitting and extraction | `autoarima.R`, `forecasts.R` |

The new scripts in `R/` make the validation contract explicit: preserve all lag evidence, correct the lag search, control false discoveries within declared families, use forward validation for forecasting and machine learning, and separate temporal precedence from structural causality.

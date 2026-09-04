# Short-Panel Temporal-Precedence Triangulation — R1

This **R1 branch is the authoritative reproducibility material** for the revised MethodsX manuscript **Triangulating Elasticity Dynamics in Short Panels with Classical, Bayesian, and Machine-Learning Precedence Tests**.

The GitHub `R1` branch and the reproducibility archive are intentionally the same repository tree. For submission or review, use GitHub **Code → Download ZIP** while viewing the `R1` branch; this avoids version drift between a separate archive and the public repository.

The workflow is intended for **wide, short panels**: repeated units observed at relatively few ordered time points, with candidate sets that may contain dozens of predictors. The R1 validation explicitly includes `K = 30` and `K = 60`; the package does not claim generic high-dimensional consistency.

> **Interpretation boundary:** this package evaluates association, predictive temporal precedence, forecasting stability, and finite-sample operating characteristics. It does not by itself identify structural causal effects.

## One-command manuscript reproduction

From the repository root:

```bash
Rscript run_reproducibility.R --mode=assets
```

This rebuilds the Specifications table, Glossary, Tables 1–8, and Figures 1–3 under `manuscript_assets/` from the audited R1 summaries in `outputs/reference/`.

For a lightweight contract check plus asset rebuild:

```bash
Rscript run_reproducibility.R --mode=quick
```

For a complete analytical recomputation:

```bash
Rscript run_reproducibility.R --mode=full --workers=10
```

Full mode reruns the several-thousand-replication R1 Monte Carlo, Dumitrescu–Hurlin and split-panel-jackknife/HPJ benchmarks, PCMCI-ParCorr benchmark, targeted no-common-factor checks, RF reference-fidelity checks, and then rebuilds the manuscript assets from the newly generated outputs. See `docs/REPRODUCIBILITY.md`.

## R1 decision contract

1. **Classical linear precedence.** Retain every lag-specific p-value, correct the lag search, then use **Benjamini–Yekutieli (BY) `q=.05` as the primary across-family rule**. Benjamini–Hochberg (BH) `q=.05` is a sensitivity result. Raw `alpha=.01` is a separate descriptive edge-level audit threshold.
2. **Bayesian linear comparison.** Store and report `log(BF10)`; the portable implementation uses a BIC approximation without clipping the log evidence.
3. **Machine-learning predictive screen.** Use forward-chaining restricted-versus-unrestricted Random Forest prediction, circular-shift shadows, the shadow 95th percentile, empirical shadow p-value, and a positive-fold-share rule. The reference setting is 500 trees and 20 shadows.
4. **Evidence-family triangulation.** Frequentist and Bayesian evidence are recorded separately but count as one **linear-model evidence family**, because R1 simulations show substantial decision dependence. RF evidence forms the second predictive/nonlinear family. The highest tier requires both families, replication, placebo success, and common-shock sensitivity.
5. **Whole-tier error interpretation.** The tier protocol is empirically calibrated in simulation. It is not presented as a theorem guaranteeing global FDR control for the full multi-branch workflow.

The smoke test explicitly protects point 4: classical + Bayesian support alone cannot be classified as triangulated temporal precedence.

## R1 operating-characteristic evidence

Primary global-null simulations (`2,000` replications per `T`) produced submitted highest-tier mean false-discovery proportions of `0.0075`, `0.0030`, and `0.0010` at `T=20,30,40`. Under the final grouped-family interpretation, the corresponding values are `0.0005`, `0`, and `0`.

Frequentist–Bayesian binary-decision phi coefficients are about `0.39–0.49` under the primary global null and `0.78–0.91` in primary partial-null cells. This dependence is why agreement between those two branches is not treated as two independent evidence families.

For the primary global null:

| T | raw alpha=.01 | raw alpha=.05 | BH q=.05 | BY q=.05 |
|---:|---:|---:|---:|---:|
| 20 | 0.0127 | 0.0529 | 0.0159 | 0.0083 |
| 30 | 0.0119 | 0.0475 | 0.0158 | 0.0073 |
| 40 | 0.0139 | 0.0534 | 0.0191 | 0.0085 |

Additional validation includes circular-shift versus non-wrapping block-surrogate calibration, `K=30/60` stress cells, variation in `N`, persistence, effect size and cross-sectional dependence, external benchmark procedures, and RF seed/tuning fidelity. The targeted 30/19 versus 500/20 RF experiment gives `93.3%` overall majority-decision agreement (`100%` for null predictors, `86.7%` for true predictors). The 500-tree/20-shadow setting remains the production default.

See `docs/METHOD_VALIDATION.md` for the reviewer-facing validation summary and `docs/ALGORITHM.md` for pseudocode, inputs/outputs, stopping rules, function signatures, and computational scaling.

## Fixed diagnostic versus final validation

Tables 4–5 and Figures 1–2 use the original fixed synthetic mixed-panel/common-driver diagnostic for implementation transparency. They are **not** used to calibrate the final R1 evidence tiers. Tables 6–8 and Figure 3 come from the reviewer-driven R1/targeted validation.

`run_demo.R` demonstrates branch mechanics only. It does **not** hard-code replication, placebo, or sensitivity gates to PASS, so it cannot manufacture the highest tier.

## Data contract

The R interface expects long data with one row per `(unit,time)` pair and explicit unit, ordered time, target, and predictor columns. Equivalent aligned matrices/tensors are used by the simulation code. Prediction preprocessing must be fitted inside each training window.

## Core R workflow

```bash
Rscript tests/smoke_test.R
Rscript run_demo.R
```

Core functions are in `R/`:

- `correlation_forecast.R`
- `classical_panel_precedence.R`
- `bayesian_panel_precedence.R`
- `rf_panel_precedence.R`
- `triangulate_evidence.R`
- `utils.R`

Reviewer-validation entry points are:

- `run_R1_validation.R`
- `run_R1_targeted_checks.R`
- `run_reproducibility.R`

The machine-readable production defaults are in `config/defaults.yml`; the full and targeted validation designs are frozen in `config/r1_validation_full.yml` and `config/r1_targeted_checks.yml`.

## Generated targets

The motivating trade application uses annual PPML/OLS elasticity estimates as second-stage targets. PPML-versus-OLS reruns are estimator-sensitivity analyses; this compact workflow does **not** claim that they propagate first-stage sampling uncertainty. Applications needing inferential propagation should implement a design-specific two-stage resampling procedure preserving the shared first-stage error structure.

## Reporting rules

- Prespecify the lag budget and testing family.
- Report raw p-values, lag-search-adjusted p-values, BY q-values, and BH sensitivity q-values separately.
- Fit restricted and unrestricted models on identical rows and controls.
- Treat full time effects and lagged leave-one-unit-out cross-sectional averages as alternative common-shock specifications unless a rank-identified design supports their joint use.
- Report `log(BF10)` rather than huge exponentiated Bayes factors.
- For RF evidence, report seed, trees, folds, mean forward MSE gain, positive-fold share, shadow count, shadow `Q0.95`, and empirical shadow p-value.
- Preserve disagreement among methods and evidence families.
- Do not interpret tier labels as structural causality or as formal global-error-rate guarantees.

## Related companion article

The empirical companion article by Z.T. Kosztyán, R. Mátrai and D. Kiss, **“Is the World Getting Smaller? Network-Driven Heterogeneity in Gravity-Model Elasticities,”** is **accepted for publication in *The World Economy***. The MethodsX method article is an independently submitted companion paper; it is **not a transfer submission**.

## Repository, archive, and license

Public repository: `https://github.com/kzst/methodsx-short-panel-triangulation`.

For R1, the repository branch itself is the reproducibility package. Select branch `R1` and choose **Code → Download ZIP** to obtain the exact same content used for review/submission.

The reference code is released under the MIT License. Cite the associated MethodsX article and software release once final bibliographic identifiers are available.

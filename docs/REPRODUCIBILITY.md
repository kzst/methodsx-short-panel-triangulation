# Reproducibility guide — MethodsX R1

This document describes the reproducibility contract for the revised MethodsX workflow **Triangulating Elasticity Dynamics in Short Panels with Classical, Bayesian, and Machine-Learning Precedence Tests**.

The GitHub `R1` branch itself is the authoritative reproducibility material. A ZIP downloaded from **Code → Download ZIP** while viewing this branch is therefore the exact same repository tree; no separate reviewer archive with a different file set is required.

## 1. What is reproduced

The material supports two distinct but complementary reproducibility tasks.

1. **Manuscript-asset reproduction.** `run_reproducibility.R --mode=assets` rebuilds every manuscript-facing CSV table and all three manuscript figures from the audited R1 summaries included in `outputs/reference/`. This mode is intentionally fast.
2. **Analytical recomputation.** `run_reproducibility.R --mode=full` reruns the several-thousand-replication R1 Monte Carlo, reviewer-requested panel-Granger and PCMCI benchmarks, the targeted no-common-factor benchmark, RF fidelity checks, and then rebuilds the manuscript assets from the newly generated outputs.

The two modes are separated because the full validation is computationally expensive, whereas a reviewer should be able to regenerate the displayed tables and figures immediately from the audited outputs used for the revision.

## 2. One-command entry point

Run from the R1 repository root.

```bash
Rscript run_reproducibility.R --mode=assets
```

This creates/refreshes:

- `manuscript_assets/tables/table_S1_specifications.csv`
- `manuscript_assets/tables/table_G1_glossary.csv`
- `manuscript_assets/tables/table_1_parameter_contract.csv`
- `manuscript_assets/tables/table_2_evidence_tiers.csv`
- `manuscript_assets/tables/table_3_troubleshooting.csv`
- `manuscript_assets/tables/table_4_known_truth_diagnostic.csv`
- `manuscript_assets/tables/table_5_common_driver_sensitivity.csv`
- `manuscript_assets/tables/table_6_operating_characteristics.csv`
- `manuscript_assets/tables/table_7_external_benchmarks.csv`
- `manuscript_assets/tables/table_8_null_calibration.csv`
- `manuscript_assets/figures/figure_1_method_specific_support.png`
- `manuscript_assets/figures/figure_2_common_driver_sensitivity.png`
- `manuscript_assets/figures/figure_3_tier_operating_characteristics.png`

A lightweight consistency run is:

```bash
Rscript run_reproducibility.R --mode=quick
```

It checks the final evidence-family contract, rebuilds the assets, and validates the frozen R1 summaries.

The full reviewer-validation run is:

```bash
Rscript run_reproducibility.R --mode=full --workers=10
```

`--workers` is optional. Full mode is intended for an ordinary checkout of the GitHub `R1` branch.

## 3. Python environment and PEP 668

The entry point uses an isolated Python virtual environment at:

```text
~/.cache/grav_methodsx_r1/venv
```

or the path supplied in the `R1_PYTHON_VENV` environment variable. It does not install packages into a Homebrew/system Python. This avoids PEP-668 `externally-managed-environment` failures on current macOS/Homebrew installations.

For manuscript assets only, the Python requirements are in `python/requirements_assets.txt`. The complete R1 validation dependencies are in `python/requirements_R1.txt`.

## 4. Final R1 decision contract

The reproducibility material implements the rule reported in the revised manuscript.

- Raw `alpha = .01` is an edge-level audit threshold, not the across-family multiplicity rule.
- Benjamini–Yekutieli at `q = .05` is primary across the declared classical testing family; BH at `q = .05` is a sensitivity result.
- Bayesian evidence is reported as `log(BF10)` rather than an exponentiated/capped Bayes factor.
- Frequentist and Bayesian branches are recorded separately but count as **one linear-model evidence family**, because their R1 decisions are substantially dependent.
- Random Forest forward-prediction evidence is the second, predictive/nonlinear evidence family.
- The highest tier requires both evidence families plus replication, placebo success, and common-shock sensitivity.
- The tier system is empirically calibrated; the material does not claim a theorem guaranteeing global FDR control for the entire multi-branch procedure.

The corrected `tests/smoke_test.R` protects this contract: classical + Bayesian support alone cannot earn the highest tier.

## 5. Production versus Monte Carlo RF settings

The user-facing/production Random Forest setting remains **500 trees and 20 circular-shift shadows**. The large Monte Carlo uses a computational approximation of **30 trees and 19 shadows**. A targeted fidelity experiment compares these settings with the same datasets and multiple seeds. The observed majority-decision agreement is 93.3% overall, 100% on null predictors, and 86.7% on true predictors. The approximation is therefore reported transparently rather than silently treated as identical to the production setting.

## 6. Frozen outputs and provenance

`outputs/reference/` contains small, audited summaries used to reconstruct the manuscript assets without rerunning the expensive validation. These are not synthetic replacements for the analysis; they are frozen extracts of the completed R1 and targeted runs.

- `final_tier_operating_characteristics.csv` — grouped-family FDP/power cells used in Table 6 and Figure 3.
- `external_benchmarks.csv` — targeted no-common-factor DH, split-panel-jackknife, and PCMCI-ParCorr results used in Table 7.
- `null_calibration_summary.csv` — circular-shift versus non-wrapping block-surrogate results used in Table 8.
- `branch_dependence_summary.csv` — classical/Bayesian dependence evidence supporting the evidence-family grouping.
- `rf_fidelity_setting_agreement.csv` — production-versus-Monte-Carlo RF fidelity.
- `demo_evidence.csv` and `common_driver_sensitivity.csv` — fixed mixed-panel implementation diagnostics used in Tables 4–5 and Figures 1–2.

When fresh full-run files exist under `outputs/R1/` and `outputs/R1_targeted/`, `python/build_manuscript_assets.py` preferentially derives Tables 6–8 and Figure 3 from those fresh outputs. The fixed mixed-panel diagnostic remains a separate implementation diagnostic, exactly as described in the revised manuscript.

## 7. Generated-target limitation

The motivating application uses annual PPML/OLS elasticity estimates as second-stage targets. This package does **not** interpret PPML-versus-OLS reruns as formal propagation of first-stage sampling uncertainty. They are estimator-sensitivity analyses. A design-specific two-stage resampling procedure is required when inferential propagation of shared first-stage estimation error is a research objective.

## 8. Table/figure mapping

| Manuscript item | Reproducibility output | Main source |
|---|---|---|
| Specifications table | `table_S1_specifications.csv` | manuscript decision contract |
| Glossary | `table_G1_glossary.csv` | manuscript decision contract |
| Table 1 | `table_1_parameter_contract.csv` | manuscript decision contract |
| Table 2 | `table_2_evidence_tiers.csv` | final grouped-family rule |
| Table 3 | `table_3_troubleshooting.csv` | manuscript protocol |
| Table 4 | `table_4_known_truth_diagnostic.csv` | fixed mixed-panel diagnostic |
| Table 5 | `table_5_common_driver_sensitivity.csv` | fixed common-driver diagnostic |
| Table 6 | `table_6_operating_characteristics.csv` | R1 grouped-family operating characteristics |
| Table 7 | `table_7_external_benchmarks.csv` | targeted benchmark outputs |
| Table 8 | `table_8_null_calibration.csv` | R1 null-calibration outputs |
| Figure 1 | `figure_1_method_specific_support.png` | fixed mixed-panel diagnostic |
| Figure 2 | `figure_2_common_driver_sensitivity.png` | fixed common-driver diagnostic |
| Figure 3 | `figure_3_tier_operating_characteristics.png` | R1 grouped-family operating characteristics |

## 9. Final consistency audit

The audited targeted benchmark gives Dumitrescu–Hurlin, `T=20`, partial-null linear power = **0.2000 (20.00%)**. A pre-audit manuscript version contained a 20.17% transcription error in this cell; the final clean and tracked R1 manuscripts have been corrected to **20.00%**, matching the authoritative output and this reproducibility material. See `docs/CONSISTENCY_AUDIT.md`. No other table-cell discrepancy was found in the programmatic comparison of Specifications, Glossary, and Tables 1–8.

## 10. Interpretation boundary

Every generated artifact concerns association, temporal precedence, predictive gain, or finite-sample operating characteristics. None of the scripts relabels these quantities as structural causal effects.

# R1 method validation for MEX-D-26-01639

The original synthetic examples are implementation diagnostics. The R1 revision adds reviewer-focused operating-characteristic validation of the evidence-tier rule and the workflow-contributed branches. Fresh full-run machine-readable results are written to `outputs/R1` and `outputs/R1_targeted`; the audited manuscript-facing extracts are committed under `outputs/reference`.

## Design

Primary cells vary `T = 20, 30, 40` with `N = 25`. The full design also varies N, K, predictor and outcome persistence, cross-sectional dependence, and planted linear/nonlinear effect size. Global-null and partial-null scenarios are both included. Primary global-null cells use 2,000 Monte Carlo replications per T and partial-null cells use 1,000 per T. Stress cells include K=30 and K=60; the claim boundary is therefore 'dozens of predictors', not a generic high-dimensional asymptotic guarantee.

The large Monte Carlo uses 30 trees/19 shadows for the RF branch for computational scalability; the user-facing reference setting remains 500 trees/20 shadows. A targeted fidelity experiment compares these directly.

## Tier operating characteristics

Under the primary global null, the mean false-discovery proportion of the submitted highest tier was 0.0075, 0.0030, and 0.0010 at T=20,30,40. After grouping frequentist and Bayesian evidence into one linear evidence family, the corresponding highest-tier values were 0.0005, 0, and 0. The top tier is therefore conservative in these simulated null settings. This is empirical operating-characteristic evidence, not a theorem establishing formal global FDR control for the whole workflow.

Under the partial null, the grouped highest tier is deliberately low-power while the replicated single-family tier retains more planted signal. The tiers are therefore interpreted as graded evidence rather than a binary discovery rule.

## Dependence among branches

The frequentist-Bayesian binary-decision phi coefficient is about 0.39-0.49 under the primary global null and about 0.78-0.91 in primary partial-null cells. Correlations involving the RF branch are much smaller. R1 therefore records the frequentist and Bayesian analyses separately but counts them as one linear evidence family in the highest tier; RF forms the predictive/nonlinear family.

## Raw alpha, BH and BY

Primary global-null edge-level rates are:

| T | raw alpha=.01 | raw alpha=.05 | BH q=.05 | BY q=.05 |
|---:|---:|---:|---:|---:|
| 20 | 0.0127 | 0.0529 | 0.0159 | 0.0083 |
| 30 | 0.0119 | 0.0475 | 0.0158 | 0.0073 |
| 40 | 0.0139 | 0.0534 | 0.0191 | 0.0085 |

`alpha=.01` is a raw/descriptive threshold. The declared across-edge testing family uses `q=.05`. Because the workflow does not assume a PRDS structure for all tested pairs, BY is primary in R1 and BH is retained as sensitivity.

## Circular-shift calibration

Targeted null simulations compare non-zero circular shifts with non-wrapping block permutations of lengths 2-4. RF circular-shift false-positive rates are about 0.04-0.06 across T=20-40, and the block alternatives are not systematically better. Circular shifts are retained as the primary negative-control construction, with block permutation reported as sensitivity. The shift is not presented as an exact exchangeability argument.

## External benchmark controls

The reviewer-requested comparator set includes Dumitrescu-Hurlin, a pooled split-panel-jackknife/HPJ Wald comparator following Juodis-Karavias-Sarafidis logic, and PCMCI-ParCorr.

In a targeted control DGP without common-factor cross-sectional dependence, global-null false-positive rates were:

| method | T=20 | T=30 | T=40 |
|---|---:|---:|---:|
| Dumitrescu-Hurlin | 0.0177 | 0.0097 | 0.0110 |
| JKS/HPJ | 0.0073 | 0.0090 | 0.0103 |
| PCMCI-ParCorr | 0.0017 | 0.0017 | 0.0013 |

JKS/HPJ is well calibrated in this assumption-compatible control environment and has high power for planted linear effects as T grows. PCMCI is conservative and strong for linear signals but weak for the planted nonlinear form used here. Under a deliberately difficult common-factor stress DGP, all external comparators can be materially mis-sized. The revision reports these as assumption-sensitive comparisons rather than claiming universal dominance of the proposed workflow.

## Finite-T classical screen

The lag-search-adjusted classical screen has global-null empirical size about 0.0119-0.0139 at raw alpha=.01; BY q=.05 reduces the edge-level false-positive rate to roughly 0.0073-0.0085. The manuscript therefore reports measured finite-T size rather than relying on the label 'directional screening' to answer small-T concerns.

## RF seed/tuning fidelity

A targeted 30-dataset-per-truth-class experiment comparing the 30/19 Monte Carlo approximation with the submitted 500/20 reference gives 100% majority-decision agreement for null predictors, 86.7% for true predictors, and 93.3% overall. The 30/19 results are treated as a conservative computational approximation; 500/20 remains the reference setting.

## Conventions

The eight-residual-df rule and the two-thirds positive-fold rule are workflow conventions rather than mathematical constants. For complete T=20,30,40 panels with lag orders 1-2, residual-df guards from 6 to 12 do not bind in the targeted calculation. RF tuning shows that the fold-share threshold can affect decisions, so applications should report sensitivity when these conventions bind.

## Generated targets

The compact R1 validation does not propagate first-stage PPML/OLS elasticity-estimation uncertainty through a design-specific two-stage bootstrap. PPML-versus-OLS reruns are estimator-sensitivity analyses, not uncertainty propagation. Applications using generated targets should state this limitation or implement a resampling scheme that preserves their first-stage error structure.

## Reproducibility

The full R1 driver is checkpointed and configuration-hashed. The targeted audit ended with status `PASS`. The full algorithm, function signatures, stopping rules, and complexity statement are documented in `docs/ALGORITHM.md`. `run_reproducibility.R --mode=assets` regenerates the displayed manuscript tables and figures from the committed audited extracts; `--mode=full` reruns the full validation before rebuilding those assets.

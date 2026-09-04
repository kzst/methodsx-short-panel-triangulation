# MethodsX R1 manuscript–reproducibility consistency audit

Audit date: 2026-09-04

## Result

**PASS. The single transcription discrepancy found during final packaging has been corrected in the final clean and tracked R1 manuscripts.**

A programmatic cell-by-cell comparison was performed between the clean R1 Word manuscript and the regenerated Specifications table, Glossary, and Tables 1–8. One benchmark transcription discrepancy was detected, corrected in both final Word manuscripts, and rechecked. The regenerated tables now match the final clean manuscript after applying the explicit publication rounding convention.

## Corrected transcription discrepancy

Table 7, Dumitrescu–Hurlin, `T=20`, partial-null linear power:

- pre-audit Word manuscript: **20.17%**
- authoritative targeted output: **20.00%**
- final clean and tracked R1 manuscripts: **20.00%**
- reproducibility output: **20.00%**

Source: the completed targeted benchmark summary reports `power_linear = 0.2` for this cell. The final manuscripts and reproducibility material now use the computed 20.00% value.

## Rounding convention fixed

The partial-null `T=30, N=25, K=6` triangulated power is `0.02975`, i.e. 2.975%. The manuscript displays **2.98%**. Python's default binary/half-even formatting can yield 2.97% in some pipelines, so the asset builder uses explicit decimal **ROUND_HALF_UP** formatting. This makes the displayed 2.98% deterministic and consistent with the manuscript.

## Decision-contract test fixed

The pre-R1 smoke test expected classical + Bayesian support to earn the highest tier. That is inconsistent with the final R1 rule because these two branches constitute one linear-model evidence family. The replacement smoke test requires both:

1. linear-model family support (frequentist and/or Bayesian); and
2. RF predictive-family support;

plus replication, placebo, and sensitivity gates for the highest tier.

## Figures

Figures 1–2 are rebuilt from the same frozen fixed-diagnostic summaries used by the manuscript. Figure 3 is rebuilt from the grouped-family R1 operating-characteristic cells. Running `Rscript run_reproducibility.R --mode=assets` regenerates all three PNGs under `manuscript_assets/figures/` from the audited data committed in this R1 branch.

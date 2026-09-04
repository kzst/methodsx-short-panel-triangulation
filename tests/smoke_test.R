# R1 decision-contract smoke test.
# This test specifically protects the grouped evidence-family interpretation
# used in the revised MethodsX manuscript.

source(file.path("R", "triangulate_evidence.R"))

# Frequentist + Bayesian agreement is ONE linear evidence family and cannot,
# by itself, earn the triangulated tier.
a <- triangulate_evidence(
  classical_supported = TRUE,
  bayesian_supported = TRUE,
  rf_supported = FALSE,
  replicated_layers = 2L,
  placebo_pass = TRUE,
  sensitivity_pass = TRUE
)
stopifnot(a$linear_family_supported, !a$predictive_family_supported)
stopifnot(a$n_evidence_families == 1L)
stopifnot(a$evidence_tier == "replicated single-family evidence")

# The highest tier requires linear-family AND predictive-family support plus
# replication, placebo, and sensitivity gates.
b <- triangulate_evidence(
  classical_supported = TRUE,
  bayesian_supported = FALSE,
  rf_supported = TRUE,
  replicated_units = 2L,
  placebo_pass = TRUE,
  sensitivity_pass = TRUE
)
stopifnot(b$n_evidence_families == 2L)
stopifnot(b$evidence_tier == "triangulated temporal precedence")

# RF-only non-replicated support is a nonlinear candidate, not triangulated.
c <- triangulate_evidence(FALSE, FALSE, TRUE)
stopifnot(c$evidence_tier == "nonlinear candidate; replication required")

cat("R1 decision-contract smoke test passed.\n")

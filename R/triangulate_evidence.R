# Evidence classification used by the MethodsX R1 workflow.
# Frequentist and Bayesian branches form one linear evidence family because
# R1 simulations show substantial decision dependence. Random Forest evidence
# forms the second predictive/nonlinear family.

triangulate_evidence <- function(classical_supported, bayesian_supported,
                                 rf_supported, replicated_units = 0L,
                                 replicated_layers = 0L,
                                 placebo_pass = FALSE,
                                 sensitivity_pass = FALSE) {
  n_methods <- sum(c(classical_supported, bayesian_supported, rf_supported),
                   na.rm = TRUE)
  linear_family_supported <- isTRUE(classical_supported) || isTRUE(bayesian_supported)
  predictive_family_supported <- isTRUE(rf_supported)
  n_evidence_families <- sum(c(linear_family_supported, predictive_family_supported))
  replicated <- replicated_units >= 2L || replicated_layers >= 2L

  if (linear_family_supported && predictive_family_supported && replicated &&
      placebo_pass && sensitivity_pass) {
    tier <- "triangulated temporal precedence"
  } else if (predictive_family_supported && !linear_family_supported && !replicated) {
    tier <- "nonlinear candidate; replication required"
  } else if (n_evidence_families >= 1L && replicated) {
    tier <- "replicated single-family evidence"
  } else if (n_evidence_families >= 1L) {
    tier <- "exploratory evidence"
  } else {
    tier <- "unsupported"
  }

  data.frame(
    n_nominal_methods = n_methods,
    linear_family_supported = linear_family_supported,
    predictive_family_supported = predictive_family_supported,
    n_evidence_families = n_evidence_families,
    replicated_units = replicated_units,
    replicated_layers = replicated_layers,
    replication_pass = replicated,
    placebo_pass = placebo_pass,
    sensitivity_pass = sensitivity_pass,
    evidence_tier = tier,
    stringsAsFactors = FALSE
  )
}

# Evidence classification. Method agreement is never converted into a
# structural-causality claim. Replication can come from repeated panel units or
# from independent data layers.

triangulate_evidence <- function(classical_supported, bayesian_supported,
                                 rf_supported, replicated_units = 0L,
                                 replicated_layers = 0L,
                                 placebo_pass = FALSE,
                                 sensitivity_pass = FALSE) {
  n_methods <- sum(c(classical_supported, bayesian_supported, rf_supported),
                   na.rm = TRUE)
  replicated <- replicated_units >= 2L || replicated_layers >= 2L
  if (n_methods >= 2L && replicated && placebo_pass && sensitivity_pass) {
    tier <- "triangulated temporal precedence"
  } else if (rf_supported && !classical_supported && !bayesian_supported) {
    tier <- "nonlinear candidate; replication required"
  } else if (n_methods >= 1L && replicated) {
    tier <- "replicated single-family evidence"
  } else if (n_methods >= 1L) {
    tier <- "exploratory evidence"
  } else {
    tier <- "unsupported"
  }
  data.frame(n_methods = n_methods,
             replicated_units = replicated_units,
             replicated_layers = replicated_layers,
             replication_pass = replicated,
             placebo_pass = placebo_pass,
             sensitivity_pass = sensitivity_pass,
             evidence_tier = tier,
             stringsAsFactors = FALSE)
}

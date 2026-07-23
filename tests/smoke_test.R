source(file.path("R", "classical_panel_precedence.R"))
source(file.path("R", "bayesian_panel_precedence.R"))
source(file.path("R", "triangulate_evidence.R"))
d <- utils::read.csv(file.path("data", "synthetic_mixed_panel.csv"))
a <- classical_panel_precedence(d, "unit", "time", "y", "x_linear",
                                  max_lag = 2, vcov = "cluster")
b <- bayesian_panel_precedence(d, "unit", "time", "y", "x_linear",
                                max_lag = 2)
stopifnot(is.finite(a$summary$p_lag_adjusted), a$summary$p_lag_adjusted < 0.05)
stopifnot(is.finite(b$summary$bf10_bic_max), b$summary$bf10_bic_max > 5)
t <- triangulate_evidence(TRUE, TRUE, FALSE, replicated_layers = 2,
                          placebo_pass = TRUE, sensitivity_pass = TRUE)
stopifnot(t$evidence_tier == "triangulated temporal precedence")
cat("Smoke test passed.\n")

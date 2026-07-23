# Minimal runnable demonstration on the supplied synthetic panel.
# Run from the repository folder: Rscript run_demo.R

source(file.path("R", "utils.R"))
source(file.path("R", "correlation_forecast.R"))
source(file.path("R", "classical_panel_precedence.R"))
source(file.path("R", "bayesian_panel_precedence.R"))
source(file.path("R", "triangulate_evidence.R"))

panel <- utils::read.csv(file.path("data", "synthetic_mixed_panel.csv"))

cor_screen <- panel_correlation_screen(
  panel, "unit", "time", c("y", "x_linear", "x_nonlinear"),
  method = "spearman", n_surrogates = 199, seed = 20260723
)
print(cor_screen)

one_unit <- panel[panel$unit == panel$unit[1L], ]
forecast_check <- rolling_origin_forecast(one_unit$y, initial_window = 12,
                                           horizon = 1, model = "naive")
print(tail(forecast_check, 3L))

classical <- classical_panel_precedence(
  panel, "unit", "time", "y", "x_linear", max_lag = 2,
  time_effects = TRUE, cross_sectional_means = FALSE, vcov = "cluster"
)
bayes <- bayesian_panel_precedence(
  panel, "unit", "time", "y", "x_linear", max_lag = 2,
  time_effects = TRUE, cross_sectional_means = FALSE
)
print(classical$summary)
print(bayes$summary)

rf_supported <- FALSE
if (requireNamespace("ranger", quietly = TRUE)) {
  source(file.path("R", "rf_panel_precedence.R"))
  rf <- rf_panel_precedence(panel, "unit", "time", "y", "x_linear",
                            max_lag = 2, trees = 300,
                            initial_window = 12, shadow_repeats = 10,
                            seed = 20260723)
  print(rf$summary)
  rf_supported <- isTRUE(rf$summary$supported)
}

classification <- triangulate_evidence(
  classical_supported = isTRUE(classical$summary$p_lag_adjusted < 0.05),
  bayesian_supported = isTRUE(bayes$summary$supported),
  rf_supported = rf_supported,
  replicated_units = 2L,
  placebo_pass = TRUE,
  sensitivity_pass = TRUE
)
print(classification)

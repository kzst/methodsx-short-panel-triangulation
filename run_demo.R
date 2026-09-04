# Minimal branch-mechanics demonstration on the supplied synthetic panel.
# Run from the repository folder: Rscript run_demo.R
#
# IMPORTANT R1 RULE:
# This compact demo does not fabricate replication/placebo/sensitivity gates.
# Consequently it cannot award the highest evidence tier by hard-coded PASS
# values. Full tier calibration is performed by run_R1_validation.R and
# run_R1_targeted_checks.R.

source(file.path("R", "utils.R"))
source(file.path("R", "correlation_forecast.R"))
source(file.path("R", "classical_panel_precedence.R"))
source(file.path("R", "bayesian_panel_precedence.R"))
source(file.path("R", "triangulate_evidence.R"))

cfg <- read_defaults()
panel <- utils::read.csv(file.path("data", "synthetic_mixed_panel.csv"))

cor_screen <- panel_correlation_screen(
  panel, "unit", "time", c("y", "x_linear", "x_nonlinear"),
  method = "spearman", n_surrogates = 199, seed = cfg$seed
)
print(cor_screen)

one_unit <- panel[panel$unit == panel$unit[1L], ]
forecast_check <- rolling_origin_forecast(
  one_unit$y,
  initial_window = cfg$forecast_initial_window,
  horizon = cfg$forecast_horizon,
  model = "naive"
)
print(tail(forecast_check, 3L))

classical <- classical_panel_precedence(
  panel, "unit", "time", "y", "x_linear", max_lag = cfg$max_lag,
  time_effects = cfg$time_effects,
  cross_sectional_means = cfg$cross_sectional_means,
  vcov = "cluster"
)
bayes <- bayesian_panel_precedence(
  panel, "unit", "time", "y", "x_linear", max_lag = cfg$max_lag,
  time_effects = cfg$time_effects,
  cross_sectional_means = cfg$cross_sectional_means,
  bf_threshold = cfg$bayes_factor_threshold
)
print(classical$summary)
print(bayes$summary)

# One edge has no across-edge testing family. Raw alpha is therefore only a
# descriptive single-edge screen here. Real candidate families use BY q=.05
# as primary and BH q=.05 as sensitivity.
classical_supported <- isTRUE(classical$summary$p_lag_adjusted <= cfg$alpha)

rf_supported <- FALSE
if (requireNamespace("ranger", quietly = TRUE)) {
  source(file.path("R", "rf_panel_precedence.R"))
  rf <- rf_panel_precedence(
    panel, "unit", "time", "y", "x_linear",
    max_lag = cfg$max_lag,
    trees = cfg$rf_trees,
    initial_window = cfg$forecast_initial_window,
    horizon = cfg$forecast_horizon,
    shadow_repeats = cfg$rf_shadow_repeats,
    positive_fold_share = cfg$rf_min_positive_fold_share,
    seed = cfg$seed
  )
  print(rf$summary)
  rf_supported <- isTRUE(rf$summary$supported)
}

# No replication/placebo/sensitivity result is manufactured in this compact
# demonstration. This protects the final R1 tier definition from the hard-coded
# gates criticized during peer review.
classification <- triangulate_evidence(
  classical_supported = classical_supported,
  bayesian_supported = isTRUE(bayes$summary$supported),
  rf_supported = rf_supported,
  replicated_units = 0L,
  replicated_layers = 0L,
  placebo_pass = FALSE,
  sensitivity_pass = FALSE
)
print(classification)
cat("For final tier operating characteristics run run_R1_validation.R and run_R1_targeted_checks.R.\n")

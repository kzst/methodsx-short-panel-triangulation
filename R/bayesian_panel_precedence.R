# Bayesian restricted-versus-unrestricted comparison for short panels.
# BIC Bayes-factor approximation is the portable default. Every candidate lag
# is reported, and prior/model sensitivity is required for substantive use.

source(file.path("R", "utils.R"))

bayesian_panel_precedence <- function(data, id, time, outcome, predictor,
                                      max_lag = 2L,
                                      unit_effects = TRUE,
                                      time_effects = TRUE,
                                      cross_sectional_means = FALSE,
                                      bf_threshold = 5) {
  assert_panel_input(data, id, time, outcome, predictor)
  if (max_lag < 1L) stop("max_lag must be positive.")
  if (time_effects && cross_sectional_means) {
    warning("Full time effects make cross-sectional means redundant; cross-sectional means are omitted. Run them as an alternative specification.")
    cross_sectional_means <- FALSE
  }
  d <- data[, unique(c(id, time, outcome, predictor)), drop = FALSE]
  d <- add_panel_lags(d, id, time, c(outcome, predictor), max_lag)
  rows <- vector("list", max_lag)

  for (lag in seq_len(max_lag)) {
    lag_y <- sprintf("%s_L%d", outcome, seq_len(lag))
    lag_x <- sprintf("%s_L%d", predictor, seq_len(lag))
    cs_cols <- character()
    if (cross_sectional_means) {
      for (nm in c(lag_y, lag_x)) {
        cs_nm <- paste0(nm, "_csmean")
        d[[cs_nm]] <- panel_cross_sectional_mean_loo(d[[nm]], d[[time]], d[[id]])
        cs_cols <- c(cs_cols, cs_nm)
      }
    }
    needed <- c(outcome, lag_y, lag_x, cs_cols, id, time)
    z <- d[stats::complete.cases(d[, needed, drop = FALSE]), , drop = FALSE]
    controls <- cs_cols
    if (unit_effects) controls <- c(controls, sprintf("factor(%s)", id))
    if (time_effects) controls <- c(controls, sprintf("factor(%s)", time))
    if (nrow(z) <= length(lag_y) + length(lag_x) + length(cs_cols) + 4L) {
      rows[[lag]] <- data.frame(lag = lag, bic_restricted = NA_real_,
                                bic_unrestricted = NA_real_, bf10_bic = NA_real_)
      next
    }
    f0 <- stats::as.formula(paste(outcome, "~", paste(c(lag_y, controls), collapse = " + ")))
    f1 <- stats::as.formula(paste(outcome, "~", paste(c(lag_y, lag_x, controls), collapse = " + ")))
    m0 <- stats::lm(f0, data = z)
    m1 <- stats::lm(f1, data = z)
    b0 <- stats::BIC(m0)
    b1 <- stats::BIC(m1)
    bf <- exp(pmin(40, pmax(-40, (b0 - b1) / 2)))
    rows[[lag]] <- data.frame(lag = lag, bic_restricted = b0,
                              bic_unrestricted = b1, bf10_bic = bf)
  }

  lag_table <- do.call(rbind, rows)
  if (!any(is.finite(lag_table$bf10_bic))) {
    best_lag <- NA_integer_
    bf_max <- NA_real_
  } else {
    best_lag <- lag_table$lag[which.max(replace(lag_table$bf10_bic,
                                                !is.finite(lag_table$bf10_bic), -Inf))]
    bf_max <- max(lag_table$bf10_bic, na.rm = TRUE)
  }
  list(
    summary = data.frame(outcome = outcome, predictor = predictor,
                         best_lag = best_lag, bf10_bic_max = bf_max,
                         supported = is.finite(bf_max) && bf_max >= bf_threshold,
                         stringsAsFactors = FALSE),
    lag_table = lag_table
  )
}

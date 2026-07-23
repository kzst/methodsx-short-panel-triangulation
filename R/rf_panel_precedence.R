# Nonlinear panel temporal-precedence screen using ranger.
# Evidence requires (i) time-blocked out-of-sample improvement over a restricted
# autoregressive model and (ii) improvement above a circular-shift shadow null.
# Shadows are created from the raw predictor and lags are rebuilt afterward.

source(file.path("R", "utils.R"))

.make_forward_folds <- function(times, initial_window, horizon = 1L) {
  u <- sort(unique(times))
  initial_window <- as.integer(initial_window)
  horizon <- as.integer(horizon)
  last_start <- length(u) - horizon + 1L
  if (initial_window < 3L || horizon < 1L ||
      last_start < initial_window + 1L) return(list())
  starts <- seq.int(initial_window + 1L, last_start)
  lapply(starts, function(k) {
    list(train = u[seq_len(k - 1L)], test = u[k:(k + horizon - 1L)])
  })
}

rf_panel_precedence <- function(data, id, time, outcome, predictor,
                                max_lag = 2L, trees = 500L,
                                initial_window = 12L, horizon = 1L,
                                shadow_repeats = 20L,
                                positive_fold_share = 0.67,
                                cross_sectional_means = TRUE,
                                time_trend = TRUE,
                                seed = 123L) {
  if (!requireNamespace("ranger", quietly = TRUE)) stop("Package 'ranger' is required.")
  assert_panel_input(data, id, time, outcome, predictor)
  if (max_lag < 1L) stop("max_lag must be positive.")
  if (shadow_repeats < 1L) stop("shadow_repeats must be positive.")
  set.seed(seed)
  raw <- data[order(data[[id]], data[[time]]),
              unique(c(id, time, outcome, predictor)), drop = FALSE]
  y_lags <- sprintf("%s_L%d", outcome, seq_len(max_lag))
  x_lags <- sprintf("%s_L%d", predictor, seq_len(max_lag))

  prepare_design <- function(x) {
    z <- add_panel_lags(x, id, time, c(outcome, predictor), max_lag)
    control_cols <- character()
    if (cross_sectional_means) {
      for (nm in c(y_lags, x_lags)) {
        cs_nm <- paste0(nm, "_csmean")
        z[[cs_nm]] <- panel_cross_sectional_mean_loo(z[[nm]], z[[time]], z[[id]])
        control_cols <- c(control_cols, cs_nm)
      }
    }
    if (time_trend) {
      tt <- match(z[[time]], sort(unique(z[[time]])))
      z$time_scaled <- (tt - min(tt)) / max(max(tt) - min(tt), 1)
      control_cols <- c(control_cols, "time_scaled")
    }
    needed <- c(outcome, y_lags, x_lags, control_cols, id, time)
    z <- z[stats::complete.cases(z[, needed, drop = FALSE]), , drop = FALSE]
    z[[id]] <- as.factor(z[[id]])
    attr(z, "control_cols") <- control_cols
    z
  }

  d <- prepare_design(raw)
  folds <- .make_forward_folds(d[[time]], initial_window, horizon)
  if (length(folds) < 3L) stop("Too few forward folds; reduce initial_window or use a longer panel.")

  score_one <- function(z, seed_offset = 0L) {
    controls <- attr(z, "control_cols")
    deltas <- numeric()
    for (fold_index in seq_along(folds)) {
      f <- folds[[fold_index]]
      tr <- z[[time]] %in% f$train
      te <- z[[time]] %in% f$test
      if (sum(te) < 2L || sum(tr) < 10L) next
      base_rhs <- c(y_lags, controls, id)
      full_rhs <- c(base_rhs, x_lags)
      base_formula <- stats::as.formula(paste(outcome, "~", paste(base_rhs, collapse = " + ")))
      full_formula <- stats::as.formula(paste(outcome, "~", paste(full_rhs, collapse = " + ")))
      m0 <- ranger::ranger(base_formula, data = z[tr, ], num.trees = trees,
                           seed = seed + seed_offset + fold_index,
                           respect.unordered.factors = "order")
      m1 <- ranger::ranger(full_formula, data = z[tr, ], num.trees = trees,
                           seed = seed + seed_offset + 1000L + fold_index,
                           respect.unordered.factors = "order")
      p0 <- predict(m0, data = z[te, ])$predictions
      p1 <- predict(m1, data = z[te, ])$predictions
      mse0 <- mean((z[[outcome]][te] - p0)^2)
      mse1 <- mean((z[[outcome]][te] - p1)^2)
      if (is.finite(mse0) && mse0 > 0 && is.finite(mse1)) {
        deltas <- c(deltas, 100 * (mse0 - mse1) / mse0)
      }
    }
    deltas
  }

  real_delta <- score_one(d)
  shadow_delta <- rep(NA_real_, shadow_repeats)
  split_idx <- split(seq_len(nrow(raw)), raw[[id]])
  for (b in seq_len(shadow_repeats)) {
    shifted <- raw
    for (idx in split_idx) shifted[[predictor]][idx] <- circular_shift_nonzero(shifted[[predictor]][idx])
    z <- prepare_design(shifted)
    tmp <- score_one(z, seed_offset = 10000L * b)
    shadow_delta[b] <- if (length(tmp)) mean(tmp) else NA_real_
  }
  avg <- if (length(real_delta)) mean(real_delta) else NA_real_
  med <- if (length(real_delta)) stats::median(real_delta) else NA_real_
  fold_share <- if (length(real_delta)) mean(real_delta > 0) else NA_real_
  valid_shadow <- shadow_delta[is.finite(shadow_delta)]
  shadow_q95 <- if (length(valid_shadow)) {
    stats::quantile(valid_shadow, 0.95, names = FALSE)
  } else {
    NA_real_
  }
  supported <- is.finite(avg) && is.finite(fold_share) && is.finite(shadow_q95) &&
    avg > 0 && avg > shadow_q95 && fold_share >= positive_fold_share
  list(
    summary = data.frame(outcome = outcome, predictor = predictor,
                         rf_delta_mse_pct_mean = avg,
                         rf_delta_mse_pct_median = med,
                         positive_fold_share = fold_share,
                         n_valid_folds = length(real_delta),
                         shadow_q95 = shadow_q95,
                         supported = supported,
                         stringsAsFactors = FALSE),
    fold_improvement = real_delta,
    shadow_distribution = shadow_delta
  )
}

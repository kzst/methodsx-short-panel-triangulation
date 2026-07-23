# Utility functions for the short-panel triangulation workflow.

`%||%` <- function(x, y) if (is.null(x)) y else x

assert_panel_input <- function(data, id, time, outcome, predictors) {
  required <- unique(c(id, time, outcome, predictors))
  missing <- setdiff(required, names(data))
  if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))
  if (anyNA(data[c(id, time)])) stop("Unit and time identifiers must not contain missing values.")
  key <- paste(data[[id]], data[[time]], sep = "\r")
  if (anyDuplicated(key)) stop("Each unit-time pair must be unique.")
  numeric_cols <- c(outcome, predictors)
  bad <- numeric_cols[!vapply(data[numeric_cols], is.numeric, logical(1))]
  if (length(bad)) stop("Outcome and predictors must be numeric: ", paste(bad, collapse = ", "))
  counts <- table(data[[id]])
  if (any(counts < 4L)) warning("At least one unit has fewer than four observations.", call. = FALSE)
  invisible(TRUE)
}

safe_scale <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

lag_vector <- function(x, k = 1L) {
  k <- as.integer(k)
  if (k < 0L) stop("k must be nonnegative")
  if (k == 0L) return(x)
  c(rep(NA, k), head(x, -k))
}

circular_shift_nonzero <- function(x) {
  n <- length(x)
  if (n <= 1L) return(x)
  k <- sample.int(n - 1L, 1L)
  c(tail(x, k), head(x, n - k))
}

add_panel_lags <- function(data, id, time, columns, max_lag) {
  max_lag <- as.integer(max_lag)
  if (!is.finite(max_lag) || max_lag < 1L) stop("max_lag must be a positive integer.")
  data <- data[order(data[[id]], data[[time]]), , drop = FALSE]
  split_idx <- split(seq_len(nrow(data)), data[[id]])
  for (nm in columns) {
    for (lag in seq_len(max_lag)) {
      out <- rep(NA_real_, nrow(data))
      for (idx in split_idx) out[idx] <- lag_vector(as.numeric(data[[nm]][idx]), lag)
      data[[sprintf("%s_L%d", nm, lag)]] <- out
    }
  }
  data
}

panel_cross_sectional_mean <- function(x, time) {
  ave(x, time, FUN = function(z) mean(z, na.rm = TRUE))
}

panel_cross_sectional_mean_loo <- function(x, time, id) {
  # Time-specific cross-sectional mean excluding the current unit. The input
  # should already be lagged, so this control uses only information available
  # before the outcome date and avoids mechanically inserting the unit's own lag.
  x <- as.numeric(x)
  total <- ave(x, time, FUN = function(z) sum(z, na.rm = TRUE))
  count <- ave(is.finite(x), time, FUN = function(z) sum(z, na.rm = TRUE))
  own <- ifelse(is.finite(x), x, 0)
  own_count <- as.integer(is.finite(x))
  denom <- count - own_count
  out <- (total - own) / denom
  out[!is.finite(out) | denom <= 0] <- NA_real_
  out
}

add_lagged_cross_sectional_means <- function(data, time, id, lag_columns) {
  for (nm in lag_columns) {
    data[[paste0("mean_", nm)]] <- panel_cross_sectional_mean_loo(
      data[[nm]], data[[time]], data[[id]]
    )
  }
  data
}

resolve_common_shock_controls <- function(time_effects, cross_sectional_means) {
  if (isTRUE(time_effects) && isTRUE(cross_sectional_means)) {
    warning(
      "Saturated time fixed effects span time-only cross-sectional means. ",
      "The cross-sectional means are omitted here; rerun with time_effects = FALSE ",
      "for a separate CCE-style sensitivity specification.",
      call. = FALSE
    )
    return(list(time_effects = TRUE, cross_sectional_means = FALSE, mode = "time fixed effects"))
  }
  if (isTRUE(time_effects)) return(list(time_effects = TRUE, cross_sectional_means = FALSE, mode = "time fixed effects"))
  if (isTRUE(cross_sectional_means)) return(list(time_effects = FALSE, cross_sectional_means = TRUE, mode = "lagged cross-sectional means"))
  list(time_effects = FALSE, cross_sectional_means = FALSE, mode = "none")
}

within_two_way <- function(x, id, time) {
  x <- as.numeric(x)
  x - ave(x, id, FUN = function(z) mean(z, na.rm = TRUE)) -
    ave(x, time, FUN = function(z) mean(z, na.rm = TRUE)) + mean(x, na.rm = TRUE)
}

unit_sign_stability <- function(x, y, id, pooled_sign) {
  idx <- split(seq_along(x), id)
  signs <- vapply(idx, function(ii) {
    ok <- is.finite(x[ii]) & is.finite(y[ii])
    if (sum(ok) < 4L) return(NA_real_)
    r <- suppressWarnings(stats::cor(x[ii][ok], y[ii][ok]))
    if (!is.finite(r) || r == 0) return(NA_real_)
    sign(r)
  }, numeric(1))
  valid <- is.finite(signs)
  if (!any(valid) || !is.finite(pooled_sign) || pooled_sign == 0) return(NA_real_)
  mean(signs[valid] == sign(pooled_sign))
}

bh_adjust_families <- function(table, p_col = "p_lag_adjusted", family_cols = c("outcome", "direction")) {
  if (!p_col %in% names(table)) stop("p_col not found")
  missing_family <- setdiff(family_cols, names(table))
  if (length(missing_family)) stop("Missing family columns: ", paste(missing_family, collapse = ", "))
  if (!length(family_cols)) {
    table$q_bh <- stats::p.adjust(table[[p_col]], method = "BH")
    return(table)
  }
  interaction_key <- interaction(table[family_cols], drop = TRUE, lex.order = TRUE)
  table$q_bh <- ave(table[[p_col]], interaction_key, FUN = function(p) {
    q <- rep(NA_real_, length(p)); ok <- is.finite(p)
    q[ok] <- stats::p.adjust(p[ok], method = "BH"); q
  })
  table
}

read_defaults <- function(path = file.path("config", "defaults.yml")) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required to read the configuration file.")
  }
  yaml::read_yaml(path)
}

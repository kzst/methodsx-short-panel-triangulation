# Classical short-panel temporal-precedence test.
# Restricted and unrestricted dynamic panel regressions are fitted on exactly
# the same rows. Every lag-specific p-value is retained; the minimum p-value is
# adjusted for the number of lags searched before across-pair FDR is applied.

source(file.path("R", "utils.R"))

classical_panel_precedence <- function(data, id, time, outcome, predictor,
                                       max_lag = 2L,
                                       unit_effects = TRUE,
                                       time_effects = TRUE,
                                       cross_sectional_means = FALSE,
                                       vcov = c("cluster", "HC1", "classical")) {
  vcov <- match.arg(vcov)
  assert_panel_input(data, id, time, outcome, predictor)
  if (max_lag < 1L) stop("max_lag must be positive.")
  if (time_effects && cross_sectional_means) {
    warning("Full time effects make cross-sectional means redundant; cross-sectional means are omitted. Run them as an alternative specification.")
    cross_sectional_means <- FALSE
  }
  d <- data[, unique(c(id, time, outcome, predictor)), drop = FALSE]
  d <- add_panel_lags(d, id, time, c(outcome, predictor), max_lag)

  tests <- vector("list", max_lag)
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
    keep <- stats::complete.cases(d[, needed, drop = FALSE])
    z <- d[keep, , drop = FALSE]
    if (nrow(z) <= length(lag_y) + length(lag_x) + length(cs_cols) + 4L) {
      tests[[lag]] <- data.frame(lag = lag, p_value = NA_real_,
                                 df_num = NA_real_, df_den = NA_real_)
      next
    }

    controls <- cs_cols
    if (unit_effects) controls <- c(controls, sprintf("factor(%s)", id))
    if (time_effects) controls <- c(controls, sprintf("factor(%s)", time))
    f0 <- stats::as.formula(paste(outcome, "~", paste(c(lag_y, controls), collapse = " + ")))
    f1 <- stats::as.formula(paste(outcome, "~", paste(c(lag_y, lag_x, controls), collapse = " + ")))
    m0 <- stats::lm(f0, data = z)
    m1 <- stats::lm(f1, data = z)

    use_robust <- vcov != "classical" && requireNamespace("sandwich", quietly = TRUE)
    if (use_robust) {
      co <- stats::coef(m1)
      keep_x <- lag_x[lag_x %in% names(co) & is.finite(co[lag_x])]
      V <- if (vcov == "cluster") {
        sandwich::vcovCL(m1, cluster = z[[id]], type = "HC1")
      } else {
        sandwich::vcovHC(m1, type = "HC1")
      }
      keep_x <- keep_x[keep_x %in% rownames(V)]
      b <- co[keep_x]
      Vs <- V[keep_x, keep_x, drop = FALSE]
      W <- tryCatch(as.numeric(t(b) %*% solve(Vs) %*% b),
                    error = function(e) NA_real_)
      df_num <- length(keep_x)
      df_den <- if (vcov == "cluster") {
        max(length(unique(z[[id]])) - 1L, 1L)
      } else {
        stats::df.residual(m1)
      }
      f_stat <- if (is.finite(W) && df_num > 0L) W / df_num else NA_real_
      p <- if (is.finite(f_stat)) stats::pf(f_stat, df_num, df_den,
                                            lower.tail = FALSE) else NA_real_
    } else {
      at <- stats::anova(m0, m1)
      p <- as.numeric(at$`Pr(>F)`[2])
      df_num <- as.numeric(at$Df[2])
      df_den <- stats::df.residual(m1)
    }
    tests[[lag]] <- data.frame(lag = lag, p_value = p,
                               df_num = df_num, df_den = df_den)
  }

  lag_table <- do.call(rbind, tests)
  ok <- is.finite(lag_table$p_value)
  if (!any(ok)) {
    best_lag <- NA_integer_
    p_min <- NA_real_
    p_lag_adjusted <- NA_real_
  } else {
    best_lag <- lag_table$lag[which.min(replace(lag_table$p_value, !ok, Inf))]
    p_min <- min(lag_table$p_value[ok])
    p_lag_adjusted <- min(1, p_min * sum(ok))
  }
  list(
    summary = data.frame(outcome = outcome, predictor = predictor,
                         best_lag = best_lag, p_min = p_min,
                         p_lag_adjusted = p_lag_adjusted,
                         n_lags_tested = sum(ok), covariance = vcov,
                         stringsAsFactors = FALSE),
    lag_table = lag_table
  )
}

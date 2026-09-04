# Correlation screening and rolling-origin forecast validation.

source(file.path("R", "utils.R"))

panel_correlation_screen <- function(data, id, time, variables,
                                     method = c("pearson", "spearman"),
                                     n_surrogates = 999L, seed = 123L) {
  method <- match.arg(method)
  if (length(variables) < 2L) stop("At least two variables are required.")
  if (n_surrogates < 1L) stop("n_surrogates must be positive.")
  assert_panel_input(data, id, time, variables[1L], variables[-1L])
  d <- data[order(data[[id]], data[[time]]), c(id, time, variables), drop = FALSE]
  set.seed(seed)
  pairs <- utils::combn(variables, 2, simplify = FALSE)
  out <- lapply(pairs, function(p) {
    x <- within_two_way(d[[p[1]]], d[[id]], d[[time]])
    y <- within_two_way(d[[p[2]]], d[[id]], d[[time]])
    obs <- suppressWarnings(stats::cor(x, y, use = "pairwise.complete.obs", method = method))
    split_idx <- split(seq_len(nrow(d)), d[[id]])
    null <- replicate(n_surrogates, {
      yp <- y
      for (idx in split_idx) yp[idx] <- circular_shift_nonzero(yp[idx])
      suppressWarnings(stats::cor(x, yp, use = "pairwise.complete.obs", method = method))
    })
    pval <- if (is.finite(obs)) {
      (1 + sum(abs(null) >= abs(obs), na.rm = TRUE)) / (n_surrogates + 1)
    } else {
      NA_real_
    }
    data.frame(var1 = p[1], var2 = p[2], correlation = obs,
               p_surrogate = pval, n_surrogates = n_surrogates,
               stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, out)
  # BY is the R1 primary multiplicity rule because arbitrary dependence among
  # tested pairs is allowed; BH is retained as a sensitivity result.
  tab$q_by <- stats::p.adjust(tab$p_surrogate, method = "BY")
  tab$q_bh <- stats::p.adjust(tab$p_surrogate, method = "BH")
  tab
}

rolling_origin_forecast <- function(y, initial_window = 12L, horizon = 1L,
                                    model = c("arima", "naive")) {
  model <- match.arg(model)
  y <- as.numeric(y)
  initial_window <- as.integer(initial_window)
  horizon <- as.integer(horizon)
  if (initial_window < 3L) stop("initial_window must be at least 3.")
  if (horizon < 1L) stop("horizon must be positive.")
  if (any(!is.finite(y))) stop("y must be finite; fit any imputation inside each training window.")
  if (length(y) < initial_window + horizon) {
    stop("Too few observations for the requested initial_window and horizon.")
  }
  origins <- seq.int(initial_window, length(y) - horizon)
  rows <- lapply(origins, function(o) {
    train <- y[seq_len(o)]
    actual <- y[o + horizon]
    if (model == "naive") {
      pred <- tail(train, 1L)
    } else {
      if (!requireNamespace("forecast", quietly = TRUE)) stop("Package 'forecast' is required.")
      fit <- forecast::auto.arima(train, seasonal = FALSE, stepwise = TRUE,
                                  approximation = FALSE)
      pred <- as.numeric(forecast::forecast(fit, h = horizon)$mean[horizon])
    }
    data.frame(origin = o, actual = actual, predicted = pred,
               error = actual - pred)
  })
  tab <- do.call(rbind, rows)
  tab$rmse_all_origins <- sqrt(mean(tab$error^2))
  tab$mae_all_origins <- mean(abs(tab$error))
  tab
}

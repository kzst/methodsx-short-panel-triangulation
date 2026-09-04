# Reviewer-requested benchmark stage for MEX-D-26-01639 R1.
# Benchmarks: Dumitrescu-Hurlin (plm::pgrangertest) and a transparent
# R implementation of the Juodis-Karavias-Sarafidis pooled HPJ logic.

`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1L]])
}
config_path <- get_arg("--config", "config/r1_validation_full.yml")
out_dir <- get_arg("--out", "outputs/R1")
workers_arg <- get_arg("--workers", "1")
workers <- suppressWarnings(as.integer(workers_arg))
if (!is.finite(workers) || workers < 1L) workers <- 1L

needed <- c("yaml", "plm", "MASS")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R package(s): ", paste(missing, collapse = ", "))
cfg <- yaml::read_yaml(config_path)
max_lag <- as.integer(cfg$max_lag %||% 2L)
q_level <- as.numeric(cfg$fdr_q %||% 0.05)
B_jks <- as.integer(cfg$classical$wild_bootstrap_reps %||% cfg$classical$hpj_bootstrap_reps %||% 199L)
seed0 <- as.integer(cfg$seed %||% 20260903L)

message(sprintf("R1 benchmark stage: max_lag=%d, JKS bootstrap=%d, workers=%d", max_lag, B_jks, workers))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
bench_dir <- file.path(out_dir, "benchmark_inputs")
if (!dir.exists(bench_dir)) stop("Benchmark input directory not found: ", bench_dir)

safe_inv <- function(A) {
  out <- tryCatch(solve(A), error = function(e) NULL)
  if (is.null(out)) MASS::ginv(A) else out
}

residual_cross <- function(y, X, Z) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(Z)) Z <- as.matrix(Z)
  if (nrow(Z) <= ncol(Z) || nrow(X) <= ncol(X)) return(NULL)
  qz <- qr(Z)
  yr <- qr.resid(qz, y)
  Xr <- apply(X, 2L, function(v) qr.resid(qz, v))
  if (is.null(dim(Xr))) Xr <- matrix(Xr, ncol = 1L)
  list(xx = crossprod(Xr), xy = crossprod(Xr, yr))
}

unit_cross_products <- function(d, y_col, x_col, p) {
  d <- d[order(d$time), , drop = FALSE]
  y <- as.numeric(d[[y_col]])
  x <- as.numeric(d[[x_col]])
  T <- length(y)
  if (T <= 2L * p + 3L) return(NULL)
  idx <- (p + 1L):T
  yy <- y[idx]
  YL <- do.call(cbind, lapply(seq_len(p), function(l) y[idx - l]))
  XL <- do.call(cbind, lapply(seq_len(p), function(l) x[idx - l]))
  if (p == 1L) {
    YL <- matrix(YL, ncol = 1L)
    XL <- matrix(XL, ncol = 1L)
  }
  Z <- cbind(1, YL)
  all_cp <- residual_cross(yy, XL, Z)
  nr <- length(yy)
  mid <- floor(nr / 2L)
  if (mid <= p + 1L || nr - mid <= p + 1L) return(NULL)
  first <- seq_len(mid)
  second <- (mid + 1L):nr
  first_cp <- residual_cross(yy[first], XL[first, , drop = FALSE], Z[first, , drop = FALSE])
  second_cp <- residual_cross(yy[second], XL[second, , drop = FALSE], Z[second, , drop = FALSE])
  if (is.null(all_cp) || is.null(first_cp) || is.null(second_cp)) return(NULL)
  list(full = all_cp, first = first_cp, second = second_cp)
}

combine_cp <- function(lst, which) {
  ok <- vapply(lst, function(z) !is.null(z), logical(1))
  lst <- lst[ok]
  if (!length(lst)) return(NULL)
  xx <- Reduce(`+`, lapply(lst, function(z) z[[which]]$xx))
  xy <- Reduce(`+`, lapply(lst, function(z) z[[which]]$xy))
  list(xx = xx, xy = xy)
}

hpj_beta_from_units <- function(cps, sampled = seq_along(cps)) {
  z <- cps[sampled]
  full <- combine_cp(z, "full")
  first <- combine_cp(z, "first")
  second <- combine_cp(z, "second")
  if (is.null(full) || is.null(first) || is.null(second)) return(NULL)
  b <- safe_inv(full$xx) %*% full$xy
  bf <- safe_inv(first$xx) %*% first$xy
  bl <- safe_inv(second$xx) %*% second$xy
  as.numeric(2 * b - (bf + bl) / 2)
}

jks_hpj_test <- function(d, y_col = "y", x_col, p = 1L, B = 199L, seed = 1L) {
  ids <- sort(unique(d$unit))
  cps <- lapply(ids, function(id) unit_cross_products(d[d$unit == id, , drop = FALSE], y_col, x_col, p))
  valid <- !vapply(cps, is.null, logical(1))
  cps <- cps[valid]
  n <- length(cps)
  if (n < 4L) return(c(p_value = NA_real_, W = NA_real_, n_units = n))
  beta <- hpj_beta_from_units(cps)
  if (is.null(beta)) return(c(p_value = NA_real_, W = NA_real_, n_units = n))
  set.seed(seed)
  draws <- matrix(NA_real_, nrow = B, ncol = length(beta))
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    draws[b, ] <- hpj_beta_from_units(cps, idx)
  }
  good <- apply(draws, 1L, function(z) all(is.finite(z)))
  draws <- draws[good, , drop = FALSE]
  if (nrow(draws) < max(30L, floor(0.8 * B))) return(c(p_value = NA_real_, W = NA_real_, n_units = n))
  V <- stats::cov(draws)
  if (length(beta) == 1L) V <- matrix(V, 1L, 1L)
  W <- as.numeric(t(beta) %*% safe_inv(V) %*% beta)
  pv <- stats::pchisq(W, df = length(beta), lower.tail = FALSE)
  c(p_value = pv, W = W, n_units = n)
}

dh_test <- function(d, y_col = "y", x_col, p = 1L) {
  z <- d[, c("unit", "time", y_col, x_col), drop = FALSE]
  names(z)[3:4] <- c("y", "x")
  pd <- plm::pdata.frame(z, index = c("unit", "time"), drop.index = FALSE)
  ans <- tryCatch(plm::pgrangertest(y ~ x, data = pd, test = "Ztilde", order = p), error = function(e) NULL)
  if (is.null(ans)) return(c(p_value = NA_real_, statistic = NA_real_))
  c(p_value = as.numeric(ans$p.value), statistic = as.numeric(ans$statistic))
}

run_rep <- function(rep_id, dat, truth, scenario, Tval) {
  d <- dat[dat$rep == rep_id, , drop = FALSE]
  pred_names <- grep("^x[0-9]+$", names(d), value = TRUE)
  K <- length(pred_names)
  dh_p <- jks_p <- rep(NA_real_, K)
  dh_stat <- jks_stat <- rep(NA_real_, K)
  for (k in seq_along(pred_names)) {
    dh <- dh_test(d, x_col = pred_names[[k]], p = max_lag)
    jk <- jks_hpj_test(d, x_col = pred_names[[k]], p = max_lag, B = B_jks,
                       seed = seed0 + rep_id * 10000L + k)
    dh_p[k] <- dh[["p_value"]]
    dh_stat[k] <- dh[["statistic"]]
    jks_p[k] <- jk[["p_value"]]
    jks_stat[k] <- jk[["W"]]
  }
  adjust_by <- function(p) {
    q <- rep(NA_real_, length(p)); ok <- is.finite(p)
    if (any(ok)) q[ok] <- p.adjust(p[ok], method = "BY")
    q
  }
  tr <- truth[truth$rep == rep_id, , drop = FALSE]
  data.frame(
    scenario = scenario, T = Tval, rep = rep_id,
    predictor = seq_len(K), truth = tr$truth[match(seq_len(K), tr$predictor)],
    signal_type = tr$signal_type[match(seq_len(K), tr$predictor)],
    dh_p = dh_p, dh_q_by = adjust_by(dh_p), dh_stat = dh_stat,
    jks_p = jks_p, jks_q_by = adjust_by(jks_p), jks_W = jks_stat,
    stringsAsFactors = FALSE
  )
}

files <- list.files(bench_dir, pattern = "^benchmark_(global_null|partial_null)_T[0-9]+\\.csv\\.gz$", full.names = TRUE)
if (!length(files)) stop("No benchmark panel files found in ", bench_dir)
all_rows <- list()
for (f in files) {
  bn <- basename(f)
  scenario <- sub("^benchmark_(global_null|partial_null)_T[0-9]+.*$", "\\1", bn)
  Tval <- as.integer(sub("^.*_T([0-9]+)\\.csv\\.gz$", "\\1", bn))
  truth_file <- sub("\\.csv\\.gz$", "_truth.csv", f)
  message("Benchmarking ", scenario, " T=", Tval)
  dat <- utils::read.csv(gzfile(f))
  truth <- utils::read.csv(truth_file, stringsAsFactors = FALSE)
  reps <- sort(unique(dat$rep))
  worker_fun <- function(rr) run_rep(rr, dat, truth, scenario, Tval)
  if (workers > 1L && .Platform$OS.type != "windows") {
    res <- parallel::mclapply(reps, worker_fun, mc.cores = workers, mc.preschedule = FALSE)
  } else {
    res <- lapply(reps, worker_fun)
  }
  all_rows[[length(all_rows) + 1L]] <- do.call(rbind, res)
}
results <- do.call(rbind, all_rows)
utils::write.csv(results, file.path(out_dir, "benchmark_panel_granger_R1.csv"), row.names = FALSE)

summary_rows <- list()
for (method in c("dh", "jks")) {
  dec <- results[[paste0(method, "_q_by")]] <= q_level
  for (key in unique(interaction(results$scenario, results$T, drop = TRUE))) {
    idx <- interaction(results$scenario, results$T, drop = TRUE) == key
    z <- results[idx, , drop = FALSE]
    d <- dec[idx]
    null <- !as.logical(z$truth)
    alt <- as.logical(z$truth)
    rep_ids <- unique(z$rep)
    fdp <- vapply(rep_ids, function(rr) {
      ii <- z$rep == rr
      ns <- sum(d[ii], na.rm = TRUE)
      if (!ns) return(0)
      sum(d[ii] & !as.logical(z$truth[ii]), na.rm = TRUE) / ns
    }, numeric(1))
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      method = method,
      scenario = z$scenario[[1]], T = z$T[[1]], n_rep = length(rep_ids),
      false_positive_rate = if (any(null)) mean(d[null], na.rm = TRUE) else NA_real_,
      power_all = if (any(alt)) mean(d[alt], na.rm = TRUE) else NA_real_,
      power_linear = if (any(z$signal_type == "linear")) mean(d[z$signal_type == "linear"], na.rm = TRUE) else NA_real_,
      power_nonlinear = if (any(z$signal_type == "nonlinear")) mean(d[z$signal_type == "nonlinear"], na.rm = TRUE) else NA_real_,
      empirical_fdr = mean(fdp, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
}
bench_summary <- do.call(rbind, summary_rows)
utils::write.csv(bench_summary, file.path(out_dir, "benchmark_panel_granger_summary_R1.csv"), row.names = FALSE)
message("R1 panel-Granger benchmark stage completed.")

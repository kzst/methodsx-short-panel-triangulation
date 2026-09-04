#!/usr/bin/env Rscript
# One-command targeted QA after the completed MEX-D-26-01639 R1 full run.

`%||%` <- function(x, y) if (is.null(x)) y else x
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  exact <- which(args == name)
  if (length(exact) && exact[[1L]] < length(args)) return(args[[exact[[1L]] + 1L]])
  pref <- grep(paste0("^", name, "="), args, value = TRUE)
  if (length(pref)) return(sub(paste0("^", name, "="), "", pref[[1L]]))
  default
}

config_path <- get_arg("--config", "config/r1_targeted_checks.yml")
workers_arg <- get_arg("--workers", "auto")
if (!file.exists(config_path)) stop("Targeted configuration not found: ", config_path)
if (!requireNamespace("yaml", quietly = TRUE)) stop("R package 'yaml' is required.")
cfg <- yaml::read_yaml(config_path)

source_out <- cfg$source_output %||% "outputs/R1"
out_dir <- cfg$output_dir %||% "outputs/R1_targeted"
if (!file.exists(file.path(source_out, "R1_MASTER_COMPLETE.json")) &&
    !file.exists(file.path(source_out, "R1_MASTER_COMPLETE.R"))) {
  stop("The completed full R1 run was not found below: ", source_out)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "R1_targeted_run.log")

log_msg <- function(...) {
  x <- paste0(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), paste(..., collapse = ""))
  cat(x, "\n")
  cat(x, "\n", file = log_file, append = TRUE)
}
run_cmd <- function(command, args = character(), label = command) {
  log_msg("START: ", label)
  status <- system2(command, args = args, stdout = "", stderr = "")
  if (!identical(status, 0L)) stop(label, " failed with exit status ", status)
  log_msg("PASS: ", label)
  invisible(status)
}

workers <- if (identical(tolower(workers_arg), "auto")) {
  max(1L, parallel::detectCores(logical = TRUE) - 1L)
} else {
  max(1L, as.integer(workers_arg))
}

Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
venv_dir <- Sys.getenv(
  "R1_PYTHON_VENV",
  unset = path.expand(file.path("~", ".cache", "grav_methodsx_r1", "venv"))
)
python <- if (.Platform$OS.type == "windows") {
  file.path(venv_dir, "Scripts", "python.exe")
} else {
  file.path(venv_dir, "bin", "python")
}
if (!file.exists(python)) stop("R1 Python virtual environment not found: ", python)
venv_check <- paste(
  "import sys;",
  "assert sys.prefix != sys.base_prefix,",
  "f'not in R1 venv: prefix={sys.prefix}, base={sys.base_prefix}'"
)
run_cmd(python, c("-c", shQuote(venv_check)), "verify existing R1 Python virtual environment")

log_msg("Targeted config: ", normalizePath(config_path, mustWork = TRUE))
log_msg("Completed source: ", normalizePath(source_out, mustWork = TRUE))
log_msg("Target output: ", out_dir, "; workers: ", workers)

run_cmd(
  python,
  c("python/r1_postprocess_existing.py", "--config", config_path,
    "--source", source_out, "--out", out_dir),
  "post-process completed full Monte Carlo"
)
run_cmd(
  python,
  c("python/r1_generate_targeted_benchmark.py", "--config", config_path, "--out", out_dir),
  "generate cross-dependence-zero benchmark controls"
)
run_cmd(
  file.path(R.home("bin"), "Rscript"),
  c("R/r1_benchmarks.R", paste0("--config=", config_path),
    paste0("--out=", out_dir), paste0("--workers=", workers)),
  "targeted Dumitrescu-Hurlin and HPJ/JKS benchmark controls"
)
if (isTRUE(cfg$benchmarks$pcmci)) {
  run_cmd(
    python,
    c("python/r1_pcmci_benchmark.py", "--config", config_path,
      "--out", out_dir, "--workers", as.character(workers)),
    "targeted PCMCI benchmark controls"
  )
}
run_cmd(
  python,
  c("python/r1_rf_reference_fidelity.py", "--config", config_path,
    "--out", out_dir, "--workers", as.character(workers)),
  "RF 30/19 versus 100/19 versus submitted 500/20 fidelity"
)
run_cmd(
  python,
  c("python/r1_targeted_finalize.py", "--config", config_path, "--out", out_dir),
  "targeted validation-gate aggregation"
)

if (!file.exists(file.path(out_dir, "R1_TARGETED_COMPLETE.json"))) {
  stop("Targeted workflow finished commands but completion marker is missing.")
}
log_msg("R1 targeted QA workflow completed. No manuscript file was modified.")

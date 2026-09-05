#!/usr/bin/env Rscript
# One-launch R1 validation driver for MEX-D-26-01639.

`%||%` <- function(x, y) if (is.null(x)) y else x
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    mode = "full",
    config = "config/r1_validation_full.yml",
    workers = NULL,
    out = NULL,
    skip_install = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (grepl("^--mode=", a)) {
      out$mode <- sub("^--mode=", "", a)
    } else if (a == "--mode" && i < length(args)) {
      i <- i + 1L
      out$mode <- args[[i]]
    } else if (grepl("^--config=", a)) {
      out$config <- sub("^--config=", "", a)
    } else if (a == "--config" && i < length(args)) {
      i <- i + 1L
      out$config <- args[[i]]
    } else if (grepl("^--workers=", a)) {
      out$workers <- sub("^--workers=", "", a)
    } else if (a == "--workers" && i < length(args)) {
      i <- i + 1L
      out$workers <- args[[i]]
    } else if (grepl("^--out=", a)) {
      out$out <- sub("^--out=", "", a)
    } else if (a == "--out" && i < length(args)) {
      i <- i + 1L
      out$out <- args[[i]]
    } else if (a == "--skip-install") {
      out$skip_install <- TRUE
    } else {
      stop("Unknown argument: ", a)
    }
    i <- i + 1L
  }
  out
}

opt <- parse_args(args)
if (!file.exists(opt$config)) stop("Configuration file not found: ", opt$config)
if (!requireNamespace("yaml", quietly = TRUE)) {
  if (opt$skip_install) stop("R package 'yaml' is required.")
  install.packages("yaml", repos = "https://cloud.r-project.org")
}

cfg <- yaml::read_yaml(opt$config)
out_dir <- opt$out %||% cfg$execution$output_dir %||% "outputs/R1"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "R1_master_run.log")

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

workers <- opt$workers
if (is.null(workers) || identical(workers, "auto")) {
  workers <- max(1L, parallel::detectCores(logical = TRUE) - 1L)
} else {
  workers <- max(1L, as.integer(workers))
}

log_msg("Configuration: ", normalizePath(opt$config, mustWork = TRUE))
log_msg("Mode: ", opt$mode, "; workers: ", workers, "; output: ", out_dir)
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")

required_r <- c("yaml", "plm", "MASS", "ranger", "sandwich", "lmtest")
missing_r <- required_r[!vapply(required_r, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_r)) {
  if (isTRUE(cfg$execution$auto_install_r_packages) && !opt$skip_install) {
    log_msg("Installing missing R packages: ", paste(missing_r, collapse = ", "))
    install.packages(missing_r, repos = "https://cloud.r-project.org", dependencies = TRUE)
  } else {
    stop("Missing R packages: ", paste(missing_r, collapse = ", "))
  }
}
missing_r <- required_r[!vapply(required_r, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_r)) stop("R dependency installation incomplete: ", paste(missing_r, collapse = ", "))

find_system_python <- function() {
  candidates <- c("python3.12", "python3.11", "python3.13", "python3", "python")
  for (nm in candidates) {
    p <- unname(Sys.which(nm))
    if (nzchar(p)) return(p)
  }
  ""
}

system_python <- find_system_python()
if (!nzchar(system_python)) stop("Python 3 was not found on PATH.")

venv_dir <- Sys.getenv(
  "R1_PYTHON_VENV",
  unset = path.expand(file.path("~", ".cache", "grav_methodsx_r1", "venv"))
)
venv_dir <- path.expand(venv_dir)
dir.create(dirname(venv_dir), recursive = TRUE, showWarnings = FALSE)
venv_python <- if (.Platform$OS.type == "windows") {
  file.path(venv_dir, "Scripts", "python.exe")
} else {
  file.path(venv_dir, "bin", "python")
}

if (!file.exists(venv_python)) {
  if (opt$skip_install) stop("Python R1 virtual environment is missing at ", venv_dir)
  run_cmd(
    system_python,
    c("-m", "venv", shQuote(venv_dir)),
    "create isolated Python R1 virtual environment"
  )
}
if (!file.exists(venv_python)) stop("Virtual environment creation failed: ", venv_dir)

# Keep the venv interpreter path; do not normalize the symlink on Homebrew/macOS.
python <- path.expand(venv_python)
log_msg("Python R1 virtual environment: ", venv_dir)
log_msg("Python interpreter (venv path): ", python)

venv_check <- paste(
  "import sys;",
  "assert sys.prefix != sys.base_prefix,",
  "f'Python is not running inside the R1 virtual environment: prefix={sys.prefix}, base={sys.base_prefix}'"
)
run_cmd(python, c("-c", shQuote(venv_check)), "verify Python R1 virtual environment")
Sys.setenv(PIP_REQUIRE_VIRTUALENV = "true")
run_cmd(python, c("-m", "pip", "--version"), "R1 virtualenv pip check")

python_check <- paste(
  "import numpy,pandas,scipy,statsmodels,sklearn,yaml;",
  if (isTRUE(cfg$benchmarks$pcmci)) "import tigramite" else "pass"
)
py_ok <- system2(
  python,
  c("-c", shQuote(python_check)),
  stdout = FALSE,
  stderr = FALSE
) == 0L

if (!py_ok) {
  if (isTRUE(cfg$execution$auto_install_python_packages) && !opt$skip_install) {
    run_cmd(
      python,
      c("-m", "pip", "install", "-r", "python/requirements_R1.txt"),
      "install Python R1 dependencies in isolated virtual environment"
    )
  } else {
    stop("Missing Python dependencies in the isolated R1 virtual environment. See python/requirements_R1.txt")
  }
}
run_cmd(python, c("-c", shQuote(python_check)), "Python dependency preflight")
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo_R1.txt"))
run_cmd(python, c("--version"), "Python version check")

if (file.exists(file.path("tests", "smoke_test.R"))) {
  run_cmd(
    file.path(R.home("bin"), "Rscript"),
    c(file.path("tests", "smoke_test.R")),
    "R1 decision-contract smoke test"
  )
}

preflight_dir <- paste0(out_dir, "_preflight")
dir.create(preflight_dir, recursive = TRUE, showWarnings = FALSE)
smoke_workers <- min(2L, workers)

run_cmd(
  python,
  c(
    "python/r1_validation.py",
    "--config", opt$config,
    "--mode", "smoke",
    "--workers", as.character(smoke_workers),
    "--out", preflight_dir
  ),
  "R1 end-to-end Python preflight"
)

run_cmd(
  file.path(R.home("bin"), "Rscript"),
  c(
    "R/r1_benchmarks.R",
    paste0("--config=", opt$config),
    paste0("--out=", preflight_dir),
    paste0("--workers=", smoke_workers)
  ),
  "R1 panel-Granger benchmark preflight"
)

if (isTRUE(cfg$benchmarks$pcmci)) {
  run_cmd(
    python,
    c(
      "python/r1_pcmci_benchmark.py",
      "--config", opt$config,
      "--out", preflight_dir,
      "--workers", as.character(smoke_workers)
    ),
    "R1 PCMCI benchmark preflight"
  )
}

run_cmd(
  python,
  c("python/r1_finalize.py", "--out", preflight_dir),
  "R1 aggregation preflight"
)
log_msg("All R1 preflight stages passed; starting requested full validation.")

py_args <- c(
  "python/r1_validation.py",
  "--config", opt$config,
  "--mode", opt$mode,
  "--workers", as.character(workers),
  "--out", out_dir
)
run_cmd(python, py_args, paste0("R1 Monte Carlo (", opt$mode, ")"))

if (opt$mode %in% c("full", "smoke")) {
  run_cmd(
    file.path(R.home("bin"), "Rscript"),
    c(
      "R/r1_benchmarks.R",
      paste0("--config=", opt$config),
      paste0("--out=", out_dir),
      paste0("--workers=", workers)
    ),
    "Dumitrescu-Hurlin and JKS-HPJ benchmarks"
  )

  if (isTRUE(cfg$benchmarks$pcmci)) {
    run_cmd(
      python,
      c(
        "python/r1_pcmci_benchmark.py",
        "--config", opt$config,
        "--out", out_dir,
        "--workers", as.character(workers)
      ),
      "PCMCI benchmark"
    )
  }

  run_cmd(
    python,
    c("python/r1_finalize.py", "--out", out_dir),
    "reviewer-facing output aggregation"
  )
}

completion <- list(
  completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  mode = opt$mode,
  workers = workers,
  config = normalizePath(opt$config, mustWork = TRUE),
  output = normalizePath(out_dir, mustWork = TRUE),
  python = python,
  python_venv = normalizePath(venv_dir, mustWork = TRUE)
)

if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(
    completion,
    file.path(out_dir, "R1_MASTER_COMPLETE.json"),
    pretty = TRUE,
    auto_unbox = TRUE
  )
} else {
  dput(completion, file = file.path(out_dir, "R1_MASTER_COMPLETE.R"))
}

log_msg("R1 validation workflow completed. No manuscript file was modified.")

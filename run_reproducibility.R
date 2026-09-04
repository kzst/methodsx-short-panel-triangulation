#!/usr/bin/env Rscript
# One-entry reproducibility driver for the MethodsX R1 branch.
#
# Modes:
#   assets : regenerate all manuscript tables and figures from audited outputs
#   quick  : run lightweight R/Python contract tests and regenerate assets
#   full   : rerun the complete R1 validation + targeted checks, then assets

args <- commandArgs(trailingOnly = TRUE)
mode <- "assets"
skip_install <- FALSE
workers <- NULL
for (i in seq_along(args)) {
  if (grepl("^--mode=", args[[i]])) mode <- sub("^--mode=", "", args[[i]])
  if (args[[i]] == "--skip-install") skip_install <- TRUE
  if (grepl("^--workers=", args[[i]])) workers <- sub("^--workers=", "", args[[i]])
}
if (!mode %in% c("assets", "quick", "full")) stop("--mode must be assets, quick, or full")

root <- normalizePath(".", mustWork = TRUE)
message("Repository root: ", root)
message("Reproducibility mode: ", mode)

run_cmd <- function(command, args = character(), label = command) {
  message("START: ", label)
  status <- system2(command, args, stdout = "", stderr = "")
  if (!identical(status, 0L)) stop(label, " failed with exit status ", status)
  message("PASS: ", label)
  invisible(status)
}

find_system_python <- function() {
  candidates <- c("python3.12", "python3.11", "python3.13", "python3", "python")
  for (nm in candidates) { p <- unname(Sys.which(nm)); if (nzchar(p)) return(p) }
  ""
}

asset_python <- function() {
  venv_dir <- path.expand(Sys.getenv("R1_PYTHON_VENV", unset = file.path("~", ".cache", "grav_methodsx_r1", "venv")))
  venv_python <- if (.Platform$OS.type == "windows") file.path(venv_dir, "Scripts", "python.exe") else file.path(venv_dir, "bin", "python")
  if (!file.exists(venv_python)) {
    if (skip_install) stop("Isolated Python environment missing at ", venv_dir, "; rerun without --skip-install or create it manually.")
    sys_py <- find_system_python()
    if (!nzchar(sys_py)) stop("Python 3 was not found. See docs/REPRODUCIBILITY.md")
    dir.create(dirname(venv_dir), recursive = TRUE, showWarnings = FALSE)
    run_cmd(sys_py, c("-m", "venv", shQuote(venv_dir)), "create isolated reproducibility Python environment")
  }
  py <- path.expand(venv_python)
  check <- paste("import sys; assert sys.prefix != sys.base_prefix,", "f'not in virtual environment: {sys.prefix} == {sys.base_prefix}'")
  run_cmd(py, c("-c", shQuote(check)), "verify isolated Python environment")
  py
}

ensure_asset_dependencies <- function(py) {
  check <- "import numpy,pandas,matplotlib"
  ok <- system2(py, c("-c", shQuote(check)), stdout = FALSE, stderr = FALSE) == 0L
  if (!ok) {
    if (skip_install) stop("Missing Python asset dependencies: numpy, pandas, matplotlib")
    Sys.setenv(PIP_REQUIRE_VIRTUALENV = "true")
    req <- file.path("python", "requirements_assets.txt")
    if (file.exists(req)) run_cmd(py, c("-m", "pip", "install", "-r", req), "install manuscript-asset Python dependencies")
    else run_cmd(py, c("-m", "pip", "install", "numpy>=1.26", "pandas>=2.2", "matplotlib>=3.8"), "install manuscript-asset Python dependencies")
  }
  run_cmd(py, c("-c", shQuote(check)), "Python manuscript-asset dependency check")
}

build_assets <- function(py) {
  run_cmd(py, c("python/build_manuscript_assets.py", "--root", root, "--out", "manuscript_assets"), "build manuscript tables and figures")
}

py <- asset_python()
ensure_asset_dependencies(py)

if (mode == "full") {
  needed <- c("run_R1_validation.R", "run_R1_targeted_checks.R")
  missing <- needed[!file.exists(needed)]
  if (length(missing)) stop("Full R1 checkout is incomplete. Missing: ", paste(missing, collapse = ", "))
  full_args <- c("run_R1_validation.R", "--mode", "full")
  if (!is.null(workers)) full_args <- c(full_args, "--workers", workers)
  if (skip_install) full_args <- c(full_args, "--skip-install")
  run_cmd(file.path(R.home("bin"), "Rscript"), full_args, "full R1 validation")
  targeted_args <- c("run_R1_targeted_checks.R")
  if (!is.null(workers)) targeted_args <- c(targeted_args, "--workers", workers)
  run_cmd(file.path(R.home("bin"), "Rscript"), targeted_args, "targeted R1 checks")
}

if (mode %in% c("quick", "full") && file.exists("tests/smoke_test.R")) {
  run_cmd(file.path(R.home("bin"), "Rscript"), "tests/smoke_test.R", "R1 evidence-family contract smoke test")
}

build_assets(py)

if (mode %in% c("quick", "full")) {
  if (file.exists("tests/test_reference_outputs.py")) run_cmd(py, "tests/test_reference_outputs.py", "reference-output contract test")
  if (file.exists("tests/test_r1_contract.py")) run_cmd(py, "tests/test_r1_contract.py", "R1 validation contract test")
}

message("Reproducibility workflow completed.")
message("Tables:  ", file.path(root, "manuscript_assets", "tables"))
message("Figures: ", file.path(root, "manuscript_assets", "figures"))

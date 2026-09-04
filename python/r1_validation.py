from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from r1_core import log
from r1_pipeline import aggregate_simulations, make_cells, run_cell
from r1_extras import (
    generate_benchmark_inputs, resolve_workers, run_null_calibration, run_tuning,
    smoke_cfg, write_gate_report, write_guard_sensitivity, write_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reviewer-driven R1 MethodsX validation engine")
    ap.add_argument("--config", default="config/r1_validation_full.yml")
    ap.add_argument(
        "--mode",
        choices=["smoke", "full", "simulate", "aggregate", "null", "tuning", "benchmark_data"],
        default="full",
    )
    ap.add_argument("--workers", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.mode == "smoke":
        cfg = smoke_cfg(cfg)
    workers = resolve_workers(args.workers if args.workers is not None else cfg["execution"]["workers"])
    out_dir = Path(args.out if args.out else cfg["execution"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, out_dir, args.mode)
    log(f"R1 validation mode={args.mode}; workers={workers}; output={out_dir}")

    if args.mode in {"smoke", "full", "simulate"}:
        cells = make_cells(cfg)
        (out_dir / "simulation_design_R1.json").write_text(
            json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8"
        )
        for cell in cells:
            run_cell(cell, cfg, out_dir, workers, chunk_size=2 if args.mode == "smoke" else 10)
    if args.mode in {"smoke", "full", "aggregate"}:
        aggregate_simulations(out_dir)
    if args.mode in {"smoke", "full", "null"}:
        run_null_calibration(cfg, out_dir, workers)
    if args.mode in {"smoke", "full", "tuning"}:
        run_tuning(cfg, out_dir, workers)
    if args.mode in {"smoke", "full", "benchmark_data"}:
        generate_benchmark_inputs(cfg, out_dir)
    if args.mode in {"smoke", "full", "aggregate"}:
        write_guard_sensitivity(cfg, out_dir)
        if (out_dir / "operating_characteristics_R1.csv").exists():
            write_gate_report(cfg, out_dir)
    log("R1 Python validation stage completed.")


if __name__ == "__main__":
    main()

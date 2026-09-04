from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def fmt(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize targeted R1 validation gates")
    ap.add_argument("--config", default="config/r1_targeted_checks.yml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out or cfg["output_dir"])
    gates = cfg["gates"]

    required = [
        "alpha_q_sensitivity_R1.csv", "tier_probability_all_R1.csv",
        "benchmark_panel_granger_summary_R1.csv", "benchmark_pcmci_summary_R1.csv",
        "rf_fidelity_summary_R1.csv", "rf_fidelity_setting_agreement_R1.csv",
    ]
    missing = [x for x in required if not (out / x).exists()]
    if missing:
        raise SystemExit("Missing targeted outputs: " + ", ".join(missing))

    bench = pd.read_csv(out / "benchmark_panel_granger_summary_R1.csv")
    jks_null = bench[(bench["method"].eq("jks")) & (bench["scenario"].eq("global_null"))]
    max_jks = float(jks_null["false_positive_rate"].max()) if len(jks_null) else np.nan
    jks_pass = bool(np.isfinite(max_jks) and max_jks <= float(gates["jks_max_global_null_fpr"]))

    agr = pd.read_csv(out / "rf_fidelity_setting_agreement_R1.csv")
    ref = str(cfg["rf_fidelity"]["reference_setting"])
    mc = agr[(agr["setting"].eq("mc_30_19")) & (agr["reference"].eq(ref))]
    mc_all = mc[mc["truth"].eq("all")]
    agreement = float(mc_all["majority_decision_agreement"].iloc[0]) if len(mc_all) else np.nan
    max_rate_diff = float(mc["decision_rate_abs_diff"].max()) if len(mc) else np.nan
    rf_pass = bool(
        np.isfinite(agreement)
        and agreement >= float(gates["rf_min_majority_agreement_with_reference"])
        and np.isfinite(max_rate_diff)
        and max_rate_diff <= float(gates["rf_max_decision_rate_abs_diff"])
    )

    aq = pd.read_csv(out / "alpha_q_sensitivity_R1.csv")
    prim = aq[aq["family"].eq("primary_global_null")].sort_values("T")

    overall = "PASS" if jks_pass and rf_pass else "TARGETED_CORRECTION_REQUIRED"
    lines = [
        "# Targeted R1 validation gates", "",
        f"Overall status: **{overall}**", "",
        "## Alpha/q post-processing", "",
        "The completed full Monte Carlo was post-processed without rerunning it. Raw alpha=0.01 and alpha=0.05 decisions are now reported separately from BH/BY q=0.05 decisions.", "",
    ]
    for _, r in prim.iterrows():
        lines.append(
            f"- T={int(r['T'])}: raw size alpha=0.01 {fmt(float(r['raw_alpha_01_false_positive_rate']))}; "
            f"raw size alpha=0.05 {fmt(float(r['raw_alpha_05_false_positive_rate']))}; "
            f"BH q=.05 {fmt(float(r['bh_q05_false_positive_rate']))}; "
            f"BY q=.05 {fmt(float(r['by_q05_false_positive_rate']))}."
        )

    lines += ["", "## Assumption-compatible JKS benchmark control", ""]
    for _, r in jks_null.sort_values("T").iterrows():
        lines.append(f"- T={int(r['T'])}: JKS/HPJ global-null false-positive rate = {fmt(float(r['false_positive_rate']))}.")
    lines.append(
        f"- Gate: {'PASS' if jks_pass else 'FAIL'}; maximum = {fmt(max_jks)}, threshold = {float(gates['jks_max_global_null_fpr']):.2f}."
    )

    lines += ["", "## RF Monte Carlo approximation fidelity", ""]
    for _, r in mc.sort_values("truth").iterrows():
        lines.append(
            f"- truth={r['truth']}: majority-decision agreement 30/19 vs submitted 500/20 = "
            f"{fmt(float(r['majority_decision_agreement']))}; decision-rate absolute difference = {fmt(float(r['decision_rate_abs_diff']))}."
        )
    lines.append(
        f"- Gate: {'PASS' if rf_pass else 'FAIL'}; overall agreement = {fmt(agreement)}, maximum truth-stratified decision-rate difference = {fmt(max_rate_diff)}."
    )

    lines += ["", "## Next action", ""]
    if overall == "PASS":
        lines.append("The reviewer-focused computational validation is complete. Proceed to one integrated manuscript/repository/response-letter revision pass; do not introduce additional analyses unless the manuscript audit uncovers a direct inconsistency.")
    else:
        if not jks_pass:
            lines.append("- Do not use the JKS benchmark results in the manuscript yet. Audit/correct the benchmark implementation and rerun only the targeted benchmark stage.")
        if not rf_pass:
            lines.append("- Do not treat the 30-tree Monte Carlo as a faithful approximation of the submitted 500-tree RF branch. Choose the closest validated setting and rerun only the affected operating-characteristic blocks before manuscript editing.")

    (out / "TARGETED_VALIDATION_GATES_R1.md").write_text("\n".join(lines), encoding="utf-8")
    complete = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": overall,
        "jks_control_pass": jks_pass,
        "jks_max_global_null_fpr": max_jks,
        "rf_fidelity_pass": rf_pass,
        "rf_30_vs_500_overall_majority_agreement": agreement,
        "rf_30_vs_500_max_decision_rate_abs_diff": max_rate_diff,
    }
    (out / "R1_TARGETED_COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(f"Targeted R1 validation completed with status: {overall}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "reference"
ASSETS = ROOT / "manuscript_assets"


def main() -> None:
    tier = pd.read_csv(REF / "final_tier_operating_characteristics.csv")
    global_null = tier[tier.scenario == "Global null"]
    partial = tier[(tier.scenario == "Partial null") & (tier.K == 6)]
    assert global_null.triangulated_fdp.max() <= 0.0005 + 1e-12
    assert partial.triangulated_fdp.max() <= 0.001 + 1e-12
    assert partial.replicated_single_family_power.max() > partial.triangulated_power.max()

    dep = pd.read_csv(REF / "branch_dependence_summary.csv")
    assert dep[dep.scenario == "Partial null"].phi_classical_by_bayes.min() > 0.75

    rf = pd.read_csv(REF / "rf_fidelity_setting_agreement.csv")
    row = rf[(rf.setting == "mc_30_19") & (rf.truth == "all")].iloc[0]
    assert abs(row.majority_decision_agreement - 0.9333333333333333) < 1e-12

    expected = [
        ASSETS / "figures" / "figure_1_method_specific_support.png",
        ASSETS / "figures" / "figure_2_common_driver_sensitivity.png",
        ASSETS / "figures" / "figure_3_tier_operating_characteristics.png",
        ASSETS / "tables" / "table_6_operating_characteristics.csv",
        ASSETS / "tables" / "table_7_external_benchmarks.csv",
        ASSETS / "tables" / "table_8_null_calibration.csv",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, f"Missing manuscript assets: {missing}"
    print("R1 reproducibility contract tests passed.")


if __name__ == "__main__":
    main()

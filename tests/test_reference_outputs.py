#!/usr/bin/env python3
"""Reference-output checks consistent with the final MethodsX R1 contract."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "reference"


def main() -> None:
    demo = pd.read_csv(REF / "demo_evidence.csv", keep_default_na=False).set_index("variable")
    # Fixed mixed-panel diagnostic: branch behavior only, not final tier logic.
    assert bool(demo.loc["x_linear", "classical_detect"])
    assert bool(demo.loc["x_linear", "bayes_detect"])
    assert bool(demo.loc["x_linear", "rf_detect"])
    assert bool(demo.loc["x_nonlinear", "rf_detect"])
    assert not bool(demo.loc["x_nonlinear", "classical_detect"])
    assert not bool(demo.loc["x_nonlinear", "bayes_detect"])
    assert not bool(demo.loc["x_null", "classical_detect"])
    assert not bool(demo.loc["x_null", "bayes_detect"])
    assert not bool(demo.loc["x_null", "rf_detect"])

    common = pd.read_csv(REF / "common_driver_sensitivity.csv").set_index("specification")
    assert float(common.loc["unit fixed effects only", "p_lag_bonf"]) < 0.05
    assert float(common.loc["unit and time fixed effects", "p_lag_bonf"]) > 0.05
    assert float(common.loc["unit effects plus lagged cross-sectional averages", "p_lag_bonf"]) > 0.05

    tier = pd.read_csv(REF / "final_tier_operating_characteristics.csv")
    g = tier[tier.scenario == "Global null"]
    assert g.triangulated_fdp.max() <= 0.0005 + 1e-12

    bench = pd.read_csv(REF / "external_benchmarks.csv")
    dh20 = bench[(bench["method"] == "Dumitrescu-Hurlin") & (bench["T"] == 20)].iloc[0]
    assert abs(float(dh20.partial_null_linear_power) - 0.20) < 1e-12
    print("Reference-output tests passed under the final R1 contract.")


if __name__ == "__main__":
    main()

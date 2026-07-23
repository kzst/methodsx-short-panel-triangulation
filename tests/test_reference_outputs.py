from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def main() -> None:
    demo = pd.read_csv(OUT / "demo_evidence.csv", keep_default_na=False)
    common = pd.read_csv(OUT / "common_driver_sensitivity.csv", keep_default_na=False)
    sim = pd.read_csv(OUT / "simulation_summary.csv", keep_default_na=False)

    by_var = demo.set_index("variable")
    assert bool(by_var.loc["x_linear", "classical_detect"])
    assert bool(by_var.loc["x_linear", "bayes_detect"])
    assert bool(by_var.loc["x_linear", "rf_detect"])
    assert "triangulated" in str(by_var.loc["x_linear", "classification"])

    assert not bool(by_var.loc["x_null", "classical_detect"])
    assert not bool(by_var.loc["x_null", "bayes_detect"])
    assert not bool(by_var.loc["x_null", "rf_detect"])

    nonlinear = by_var.loc["x_nonlinear"]
    assert bool(nonlinear["rf_detect"])
    assert not bool(nonlinear["classical_detect"])
    assert not bool(nonlinear["bayes_detect"])

    common = common.set_index("specification")
    assert float(common.loc["unit fixed effects only", "p_lag_bonf"]) < 0.05
    assert float(common.loc["unit and time fixed effects", "p_lag_bonf"]) > 0.05
    assert float(common.loc["unit effects plus lagged cross-sectional averages", "p_lag_bonf"]) > 0.05

    assert len(sim) == 24
    assert set(sim["scenario"]) == {"null", "linear", "nonlinear", "break"}
    assert set(sim["n_time"].astype(int)) == {20, 28, 40}
    assert set(sim["n_rep"].astype(int)) == {100}

    print("Python reference-output tests passed.")

if __name__ == "__main__":
    main()

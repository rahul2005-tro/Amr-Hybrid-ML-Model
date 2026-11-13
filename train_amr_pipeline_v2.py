"""Train a weighted AMR runtime ensemble on reproducible synthetic telemetry.

Run with Python 3.13:
    python train_arm_pipeline.py

Outputs go to ensemble_artifacts/ beside this script. Existing source files
and the original amr_pipeline.joblib are not modified.
"""
from pathlib import Path
import argparse
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm
import catboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def generate_telemetry():
    """Reproduce the original 8,000-row simulation with seed 42."""
    rng = np.random.default_rng(42)
    n = 8000
    terrain_types = rng.choice(["Smooth_Epoxy", "Textured_Concrete", "Outdoor_Asphalt"], size=n, p=[0.5, 0.35, 0.15])
    terrain_factor = np.where(terrain_types == "Smooth_Epoxy", 1.0, np.where(terrain_types == "Textured_Concrete", 1.18, 1.38))

    soc_pct = rng.uniform(12.0, 100.0, size=n)
    payload_kg = rng.uniform(0.0, 120.0, size=n)
    velocity_ms = rng.uniform(0.3, 2.0, size=n)
    incline_deg = rng.choice([0.0, 2.0, 5.0, 8.0, 12.0], size=n, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    ambient_temp_c = rng.normal(28.0, 5.0, size=n).clip(15.0, 45.0)
    battery_cycles = rng.integers(10, 800, size=n)

    total_mass = 65.0 + payload_kg
    incline_rad = np.radians(incline_deg)
    current_a = (3.5 + (0.08 * total_mass * velocity_ms * terrain_factor) + 
                 (0.35 * total_mass * np.sin(incline_rad) * velocity_ms) + rng.normal(0, 0.8, n)).clip(2.5, 65.0)
    internal_res = 0.015 * (1 + (battery_cycles / 1000.0))
    voltage_v = (22.0 + (0.038 * soc_pct) - (current_a * internal_res) + rng.normal(0, 0.08, n)).clip(20.5, 26.8)
    cell_temp_c = ambient_temp_c + (0.12 * (current_a ** 1.6) * internal_res * 15) + rng.normal(0, 0.6, n)

    usable_energy_wh = 960.0 * (1 - 0.00018 * battery_cycles) * ((soc_pct - 10.0) / 100.0)
    thermal_derating = np.where(cell_temp_c > 42.0, 0.82, 1.0)
    power_draw_w = voltage_v * current_a
    remaining_runtime_min = ((usable_energy_wh / power_draw_w) * 60.0 * thermal_derating + rng.normal(0, 1.2, n)).clip(1.0, 300.0)

    df = pd.DataFrame({
        "Terrain_Surface": terrain_types,
        "SoC [%]": np.round(soc_pct, 1),
        "Voltage [V]": np.round(voltage_v, 2),
        "Current [A]": np.round(current_a, 2),
        "Payload [kg]": np.round(payload_kg, 1),
        "Velocity [m/s]": np.round(velocity_ms, 2),
        "Incline [deg]": incline_deg,
        "Ambient_Temp [C]": np.round(ambient_temp_c, 1),
        "Cell_Temp [C]": np.round(cell_temp_c, 1),
        "Battery_Cycles": battery_cycles,
        "Remaining_Runtime [min]": np.round(remaining_runtime_min, 1)
    })
    return df


def prepare_split(telemetry):
    """Preserve the 15 input features expected by the existing application."""
    df = telemetry.copy()
    df["Terrain_Surface"] = df["Terrain_Surface"].map({"Smooth_Epoxy": 0, "Textured_Concrete": 1, "Outdoor_Asphalt": 2})
    df["Power_Draw [W]"] = df["Voltage [V]"] * df["Current [A]"]
    df["Temp_Rise [C]"] = df["Cell_Temp [C]"] - df["Ambient_Temp [C]"]
    df["Mechanical_Load_Index"] = (65.0 + df["Payload [kg]"]) * df["Velocity [m/s]"] * (1.0 + np.sin(np.radians(df["Incline [deg]"])))
    df["Available_Energy [Wh]"] = 960.0 * ((df["SoC [%]"] - 10.0) / 100.0)
    df["Theoretical_Runtime [min]"] = (df["Available_Energy [Wh]"] / df["Power_Draw [W]"]) * 60.0

    X = df.drop(columns=["Remaining_Runtime [min]"])
    y = df["Remaining_Runtime [min]"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


def build_pipeline():
    xgb_model = XGBRegressor(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        random_state=42, n_jobs=-1,
    )
    lgbm_model = LGBMRegressor(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    cat_model = CatBoostRegressor(
        iterations=300, learning_rate=0.08, depth=6,
        random_seed=42, verbose=0, thread_count=-1,
        allow_writing_files=False,
    )
    # Explicit NumPy output avoids passing bracketed feature names to XGBoost.
    scaler = StandardScaler().set_output(transform="default")
    return Pipeline([
        ("scaler", scaler),
        ("ensemble", VotingRegressor(
            estimators=[("xgb", xgb_model), ("lgbm", lgbm_model), ("cat", cat_model)],
            weights=[2, 2, 1], n_jobs=-1,
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "ensemble_artifacts",
        help="Destination for the new model, telemetry and evaluation files.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    filenames = ["amr_pipeline.joblib", "amr_battery_telemetry.csv", "metrics.json", "test_predictions.csv"]
    # Fail before training if the destination would replace any existing artifact.
    existing = [str(output / name) for name in filenames if (output / name).exists()]
    if existing:
        parser.error("Output files already exist; choose a new --output-dir: " + ", ".join(existing))

    telemetry = generate_telemetry()
    X_train, X_test, y_train, y_test = prepare_split(telemetry)
    amr_pipeline = build_pipeline()
    print(f"Training on {len(X_train):,} synthetic samples; testing on {len(X_test):,}.", flush=True)
    amr_pipeline.fit(X_train, y_train)
    predictions = amr_pipeline.predict(X_test)
    if not np.isfinite(predictions).all():
        raise ValueError("The ensemble produced non-finite predictions.")
    metrics = {
        "mae_minutes": float(mean_absolute_error(y_test, predictions)),
        "rmse_minutes": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "random_seed": 42, "data_source": "synthetic physics simulation",
        "ensemble_weights": {"xgb": 2, "lgbm": 2, "cat": 1},
        "features": list(X_train.columns),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                     "catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "amr_pipeline.joblib"
    joblib.dump(amr_pipeline, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored.predict(X_test.iloc[:20]), predictions[:20])
    telemetry.to_csv(output / "amr_battery_telemetry.csv", index=False)
    pd.DataFrame({"source_row": X_test.index, "actual_minutes": y_test.to_numpy(),
                  "predicted_minutes": predictions}).to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"MAE: {metrics['mae_minutes']:.3f} min | RMSE: {metrics['rmse_minutes']:.3f} min | R²: {metrics['r2']:.5f}")
    print(f"Saved upgraded 3-model voting ensemble to {model_path}")
    print("Verified that saved-model predictions match in-memory predictions.")


if __name__ == "__main__":
    main()

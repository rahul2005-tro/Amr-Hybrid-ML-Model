# AMR Hybrid Studio

A separate Streamlit dashboard for the fitted **XGBoost + LightGBM + CatBoost** voting ensemble. Model weights are **2:2:1**, equivalent to 40%, 40%, and 20%. The existing single-model dashboard is unchanged.

## Run

Use Python 3.13. From this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py --server.port=8503
```

Or run `./run.ps1` in PowerShell. The launcher binds to localhost. On Linux/macOS use `python3.13` and `.venv/bin/python`.

## Dashboard

- Mission analysis: runtime envelope, electrical and thermal estimates, mission budget, and sensitivity charts.
- Inside the ensemble: each fitted member's actual prediction, weighted contribution, and CSV export.
- Scenario comparison: save up to six mission snapshots in the current session.
- Dataset & validation: held-out predictions, residuals, dataset filters, and per-member feature importance.

The bundled model, data, metrics, and test predictions were copied from `ensemble_artifacts`. No retraining is performed on app startup. All runtime inference uses the saved fitted scaler and ensemble.

## Evaluation

The supplied 6,400/1,600 synthetic train/test split uses seed 42. Bundled results: MAE 1.1948 min, RMSE 1.7316 min, R² 0.998982. Run the checks to reproduce those metrics against the bundled model:

```powershell
.\.venv\Scripts\python test_dashboard.py
```

The checks also verify weighted averaging, saved prediction parity, input feature equations, boundary inference, mission presets, scenario save/clear, and empty dataset filters.

## Interpretation

This is a physics-informed simulation, not live telemetry or a field-validated BMS. Runtime is predicted by the ensemble; voltage, current, and cell temperature use the original physical approximations. Range assumes continuous constant-speed motion. MAE and member disagreement are not prediction intervals. Feature importance definitions differ across libraries and are displayed separately rather than averaged.

## Files and sharing

Keep `app.py`, `engine.py`, `amr_pipeline.joblib`, `amr_battery_telemetry.csv`, `metrics.json`, `test_predictions.csv`, `requirements.txt`, and `.streamlit/config.toml` together. The folder can be its own GitHub repository. The included `.gitignore` excludes the virtual environment and local logs. Hosting requires a Python-capable Streamlit server; the entry point is `app.py`.

`train_amr_pipeline_v2.py` is included for reproducibility. Running it creates a new `ensemble_artifacts` subfolder; it does not automatically replace the model used by this dashboard. Only load trusted joblib artifacts.

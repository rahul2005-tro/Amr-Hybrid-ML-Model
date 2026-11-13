"""Run with python test_dashboard.py from the project directory."""
from pathlib import Path
import ast
import itertools
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from engine import Mission, features, evaluate, model_votes
import json
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from train_amr_pipeline_v2 import prepare_split

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "amr_pipeline.joblib")
assert list(features(Mission()).columns) == list(model.feature_names_in_)
votes = model_votes(model, Mission())
assert np.allclose(votes.Weight, [.4, .4, .2])
assert np.isclose(votes['Contribution (min)'].sum(), evaluate(model, Mission())['Runtime [min]'])
_, held_out, _, targets = prepare_split(pd.read_csv(ROOT / 'amr_battery_telemetry.csv'))
predictions = model.predict(held_out)
report = json.loads((ROOT / 'metrics.json').read_text())
for key, value in [('mae_minutes', mean_absolute_error(targets, predictions)),
                   ('rmse_minutes', root_mean_squared_error(targets, predictions)),
                   ('r2', r2_score(targets, predictions))]:
    assert np.isclose(report[key], value), (key, value)
saved = pd.read_csv(ROOT / 'test_predictions.csv')
np.testing.assert_allclose(saved.predicted_minutes, predictions)

# Compare feature values to the actual original app when it is available.
original = ROOT.parent / "app.py"
if original.exists():
    source = original.read_text(encoding="utf-8")
    section = source[source.index("total_mass ="):source.index("pred_runtime_min =")]
    m = Mission()
    scope = dict(np=np, pd=pd, payload_kg=m.payload, velocity_ms=m.speed,
        incline_deg=m.incline, battery_cycles=m.cycles, soc_pct=m.soc,
        ambient_temp_c=m.ambient, terrain_code=0, terrain_factor=1.)
    exec(compile(ast.parse(section), "original-feature-equations", "exec"), scope)
    pd.testing.assert_frame_equal(scope["input_df"], features(m), check_dtype=False)

for payload, speed, soc, ambient in itertools.product([0.,120.],[.3,2.],[15.,100.],[15.,45.]):
    result = evaluate(model, Mission(payload=payload, speed=speed, soc=soc, ambient=ambient))
    assert np.isfinite(list(result.values())).all()
    assert result["Runtime [min]"] > 0

app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
assert not app.exception, app.exception
app.button[0].click().run()
assert len(app.session_state["scenarios"]) == 1
app.selectbox(key="preset").select("Heavy outdoor run").run()
assert not app.exception
assert app.session_state["payload"] == 100
assert any("thermal" in w.value.lower() for w in app.warning)
app.button[1].click().run()
assert app.session_state["scenarios"] == []
app.multiselect[0].set_value([]).run()
assert not app.exception
print("PASS: weighted votes, reproduced held-out metrics/predictions, feature parity, boundary predictions, app render, presets, scenario save/clear, empty filter")

"""Physics feature construction, kept equivalent to the original application."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

TERRAINS = {"Smooth epoxy": (0, 1.0), "Textured concrete": (1, 1.18), "Outdoor asphalt": (2, 1.38)}

@dataclass(frozen=True)
class Mission:
    terrain: str = "Smooth epoxy"
    payload: float = 50.0
    speed: float = 1.2
    incline: float = 2.0
    soc: float = 75.0
    cycles: int = 250
    ambient: float = 28.0

def features(m: Mission) -> pd.DataFrame:
    code, factor = TERRAINS[m.terrain]
    mass, slope = 65 + m.payload, np.radians(m.incline)
    current = np.clip(3.5 + .08 * mass * m.speed * factor + .35 * mass * np.sin(slope) * m.speed, 2.5, 65)
    resistance = .015 * (1 + m.cycles / 1000)
    voltage = np.clip(22 + .038 * m.soc - current * resistance, 20.5, 26.8)
    temperature = m.ambient + .12 * current ** 1.6 * resistance * 15
    power, energy = voltage * current, 960 * (m.soc - 10) / 100
    return pd.DataFrame([{"Terrain_Surface": code, "SoC [%]": m.soc, "Voltage [V]": voltage,
        "Current [A]": current, "Payload [kg]": m.payload, "Velocity [m/s]": m.speed,
        "Incline [deg]": m.incline, "Ambient_Temp [C]": m.ambient, "Cell_Temp [C]": temperature,
        "Battery_Cycles": m.cycles, "Power_Draw [W]": power, "Temp_Rise [C]": temperature - m.ambient,
        "Mechanical_Load_Index": mass * m.speed * (1 + np.sin(slope)),
        "Available_Energy [Wh]": energy, "Theoretical_Runtime [min]": energy / power * 60}])

def evaluate(model, mission):
    frame = features(mission)
    result = frame.iloc[0].to_dict()
    result["Runtime [min]"] = float(model.predict(frame)[0])
    result["Range [km]"] = result["Runtime [min]"] * mission.speed * .06
    return result

def sweep(model, mission, field, values):
    frame = pd.concat([features(replace(mission, **{field: float(v)})) for v in values], ignore_index=True)
    return pd.DataFrame({field: values, "Model": model.predict(frame), "Physics baseline": frame["Theoretical_Runtime [min]"]})

def model_votes(model, mission):
    """Use the fitted scaler and fitted members, never refit at inference time."""
    scaled = model.named_steps["scaler"].transform(features(mission))
    ensemble = model.named_steps["ensemble"]
    names = [name for name, _ in ensemble.estimators]
    predictions = [float(member.predict(scaled)[0]) for member in ensemble.estimators_]
    weights = np.asarray(ensemble.weights, dtype=float)
    weights /= weights.sum()
    return pd.DataFrame({"Model": [{"xgb":"XGBoost", "lgbm":"LightGBM", "cat":"CatBoost"}[n] for n in names],
        "Runtime (min)": predictions, "Weight": weights,
        "Contribution (min)": np.asarray(predictions) * weights})

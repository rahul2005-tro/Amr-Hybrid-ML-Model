from pathlib import Path
from dataclasses import asdict
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import Mission, TERRAINS, evaluate, sweep, model_votes

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AMR / Hybrid Studio", page_icon="⬡", layout="wide")
st.markdown("""<style>
.block-container{padding:2rem;max-width:1500px}
@media(max-width:1100px){[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}[data-testid="stColumn"]{min-width:min(230px,100%)!important;flex:1 1 230px!important}}
h1,h2,h3{letter-spacing:-.035em}h1{font-weight:650!important;font-size:2.6rem!important}
[data-testid="stSidebar"]{border-right:1px solid #dfe5e8;background:#fff}
[data-testid="stSidebar"] .block-container{padding-top:1.5rem}
.eyebrow{font:600 11px monospace;letter-spacing:.16em;color:#68808b;text-transform:uppercase}
.brand{font-size:23px;font-weight:750;letter-spacing:-1px;color:#182a39;margin-bottom:6px}
.brand span{color:#35658c}.subtle{color:#73828d;font-size:13px}
.metric{background:#fff;border:1px solid #dfe5e8;border-radius:9px;padding:20px 22px;min-height:147px}
.metric.primary{background:#203c56;border-color:#203c56;color:#fff}
.metric .label{font-size:12px;letter-spacing:.04em;color:#6a7e89}
.metric.primary .label,.metric.primary .note{color:#b5cfca}
.metric .value{font-size:36px;font-weight:650;letter-spacing:-1.6px;margin:9px 0 5px;line-height:1.2;white-space:nowrap}
.metric .unit{font-size:15px;letter-spacing:0;font-weight:400;margin-left:5px}
.metric .note{font-size:12px;color:#6a7e89}
.status{padding:14px 18px;border-left:3px solid #35658c;background:#e9f2ee;margin:20px 0;color:#28594e;font-size:14px}
.status.warn{background:#fcf1e3;border-color:#b27931;color:#815720}
.foot{border-top:1px solid #dfe5e8;margin-top:28px;padding-top:16px;color:#73828d;font:11px monospace}
[data-testid="stMetricValue"]{font-size:25px}
div[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:10px}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    pipeline = joblib.load(ROOT / "amr_pipeline.joblib")
    if "ensemble" not in pipeline.named_steps:
        raise ValueError("This dashboard requires the fitted voting ensemble.")
    return pipeline

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "amr_battery_telemetry.csv")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model. Install requirements.txt and keep amr_pipeline.joblib beside app.py. Details: {exc}")
    st.stop()

PRESETS = {"Standard delivery": Mission(), "Light indoor run": Mission(payload=20, speed=.8, incline=0, soc=90),
           "Heavy outdoor run": Mission(terrain="Outdoor asphalt", payload=100, speed=1.5, incline=5, soc=60, cycles=500, ambient=35)}
def set_preset():
    for key, value in asdict(PRESETS[st.session_state.preset]).items():
        st.session_state[key] = value
for key, value in asdict(Mission()).items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.markdown('<div class="brand">⬡ AMR<span> / </span>HYBRID</div><div class="subtle">Three models. One mission estimate.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="eyebrow">01 / Mission profile</div>', unsafe_allow_html=True)
    st.selectbox("Load a starting profile", list(PRESETS), key="preset", on_change=set_preset)
    st.selectbox("Floor surface", list(TERRAINS), key="terrain")
    st.slider("Payload · kg", 0.0, 120.0, step=2.5, key="payload")
    st.slider("Travel speed · m/s", .3, 2.0, step=.1, key="speed")
    st.select_slider("Ramp incline · degrees", [0., 2., 5., 8., 12.], key="incline")
    st.divider()
    st.markdown('<div class="eyebrow">02 / Battery conditions</div>', unsafe_allow_html=True)
    st.slider("State of charge · %", 15., 100., step=1., key="soc")
    st.slider("Battery age · cycles", 10, 800, step=10, key="cycles")
    st.slider("Ambient temperature · °C", 15., 45., step=.5, key="ambient")
    st.caption("24 V · 40 Ah · LiFePO₄\n\n65 kg platform · 10% discharge floor")

mission = Mission(**{k: st.session_state[k] for k in asdict(Mission())})
r = evaluate(model, mission)
st.markdown('<div class="eyebrow">Ensemble workspace / 02</div>', unsafe_allow_html=True)
title, badge = st.columns([4, 1])
with title:
    st.title("Hybrid mission analysis")
    st.caption("Explore the operating envelope of your autonomous mobile robot.")
with badge:
    st.markdown('<div style="text-align:right;padding-top:24px" class="eyebrow">● Simulation mode<br><br>Weighted ensemble · 2:2:1</div>', unsafe_allow_html=True)
st.write("")
metrics = [("PREDICTED RUNTIME", r["Runtime [min]"], "min", f'{r["Runtime [min]"]-r["Theoretical_Runtime [min]"]:+.1f} min vs. ideal physics', True),
           ("CONTINUOUS TRAVEL", r["Range [km]"], "km", f'At a constant {mission.speed:.1f} m/s', False),
           ("ELECTRICAL LOAD", r["Power_Draw [W]"], "W", f'{r["Voltage [V]"]:.1f} V / {r["Current [A]"]:.1f} A', False),
           ("ESTIMATED CELL TEMP", r["Cell_Temp [C]"], "°C", f'{r["Temp_Rise [C]"]:.1f} °C above ambient', False)]
for col, (label, value, unit, note, primary) in zip(st.columns(4), metrics):
    col.markdown(f'<div class="metric {"primary" if primary else ""}"><div class="label">{label}</div><div class="value">{value:.1f}<span class="unit">{unit}</span></div><div class="note">{note}</div></div>', unsafe_allow_html=True)

thermal = r["Cell_Temp [C]"] > 42
message = "Thermal threshold exceeded. The training simulation applies 18% capacity derating above 42 °C." if thermal else "Below the simulation’s 42 °C thermal threshold. Check the mission budget before dispatch."
st.markdown(f'<div class="status {"warn" if thermal else ""}">{message}</div>', unsafe_allow_html=True)

def chart_style(fig, height=310):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=25, b=15), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#647985", size=12),
        legend=dict(orientation="h", y=1.18, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e6ebee", zeroline=False)
    return fig

overview, ensemble_tab, comparison, data_tab = st.tabs(["Mission analysis", "Inside the ensemble", "Scenario comparison", "Dataset & validation"])
with overview:
    left, right = st.columns([1.8, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Charge-to-runtime envelope")
        st.caption("Model predictions across starting charge levels; all other inputs held fixed.")
        curve = sweep(model, mission, "soc", np.linspace(15, 100, 45))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.soc, y=curve.Model, name="Hybrid runtime", line=dict(color="#35658c", width=3), fill="tozeroy", fillcolor="rgba(53,101,140,.07)"))
        fig.add_trace(go.Scatter(x=curve.soc, y=curve["Physics baseline"], name="Ideal physics", line=dict(color="#a2adb5", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=[mission.soc], y=[r["Runtime [min]"]], mode="markers", name="Current mission", marker=dict(size=11,color="#d38c40",line=dict(color="white",width=2))))
        fig.update_layout(xaxis_title="Starting state of charge (%)", yaxis_title="Runtime (min)")
        st.plotly_chart(chart_style(fig), width="stretch")
    with right, st.container(border=True):
        st.subheader("Mission budget")
        st.caption("Plan a duration and leave an operational buffer.")
        duration = st.number_input("Planned mission · minutes", 5, 300, 60, 5)
        reserve = st.slider("Runtime buffer · %", 0, 40, 15, 5)
        usable = r["Runtime [min]"] * (1 - reserve / 100)
        margin = usable - duration
        st.metric("Budget remaining", f"{margin:+.1f} min")
        st.progress(min(1., max(0., duration / max(usable, .01))))
        if thermal:
            st.warning("Review thermal conditions before planning this run.")
        elif margin < 0:
            st.warning("Mission exceeds the buffered runtime estimate.")
        else:
            st.success("Within the selected runtime budget.")
        st.caption(f"{usable:.1f} min available after a {reserve}% buffer. This is a planning estimate, not a safety certification.")
    with st.container(border=True):
        label, control = st.columns([3, 1])
        label.subheader("What changes the runtime?")
        axis = control.selectbox("Explore a variable", ["Payload", "Travel speed", "Ambient temperature"], label_visibility="collapsed")
        field, values, unit = {"Payload": ("payload", np.linspace(0,120,31), "kg"), "Travel speed": ("speed", np.linspace(.3,2,35), "m/s"), "Ambient temperature": ("ambient",np.linspace(15,45,31),"°C")}[axis]
        s = sweep(model, mission, field, values)
        f = go.Figure(go.Scatter(x=s[field], y=s.Model, line=dict(color="#35658c",width=3), name="Predicted runtime"))
        f.add_vline(x=getattr(mission, field), line_dash="dot", line_color="#d38c40")
        f.update_layout(xaxis_title=f"{axis} ({unit})", yaxis_title="Runtime (min)")
        st.plotly_chart(chart_style(f, 245), width="stretch")
        st.caption("Dotted line marks the current setting. One-variable model sensitivity; curves are not measured discharge traces.")

with ensemble_tab:
    st.subheader("Three estimates. One weighted decision.")
    st.caption("Each fitted model sees the same scaled mission inputs. The final prediction is their weighted average.")
    votes = model_votes(model, mission)
    for column, row in zip(st.columns(3), votes.to_dict("records")):
        with column, st.container(border=True):
            st.markdown(f"**{row['Model']}**")
            st.metric("Member estimate", f"{row['Runtime (min)']:.2f} min")
            st.caption(f"{row['Weight']:.0%} weight · {row['Contribution (min)']:.2f} min contribution")
    st.info(f"Hybrid runtime: {votes['Contribution (min)'].sum():.2f} min = 0.4 × XGBoost + 0.4 × LightGBM + 0.2 × CatBoost")
    fig = go.Figure(go.Bar(x=votes["Runtime (min)"], y=votes.Model, orientation="h",
        marker_color=["#35658c", "#639b8a", "#c49752"], text=votes["Runtime (min)"].round(2), textposition="auto"))
    fig.add_vline(x=r["Runtime [min]"], line_dash="dot", line_color="#203c56")
    fig.update_layout(xaxis_title="Predicted runtime (min)")
    st.plotly_chart(chart_style(fig, 270), width="stretch")
    st.caption(f"Member spread: {votes['Runtime (min)'].max()-votes['Runtime (min)'].min():.2f} min. Model disagreement is not a calibrated uncertainty interval.")
    st.download_button("Export member predictions", votes.to_csv(index=False), "hybrid-votes.csv", "text/csv")

with comparison:
    st.subheader("Compare operating plans")
    st.caption("Save up to six input snapshots, then compare their model estimates.")
    st.session_state.setdefault("scenarios", [])
    c1, c2, c3 = st.columns([3,1,1])
    name = c1.text_input("Scenario name", value=f"Mission {len(st.session_state.scenarios)+1}", max_chars=50)
    if c2.button("Save current", use_container_width=True):
        if len(st.session_state.scenarios) >= 6:
            st.warning("Six scenarios saved. Clear the list to start another comparison.")
        elif not name.strip():
            st.warning("Enter a scenario name.")
        else:
            st.session_state.scenarios.append({"Scenario": name.strip(), **asdict(mission), "Runtime (min)": round(r["Runtime [min]"],2), "Range (km)": round(r["Range [km]"],2), "Cell temp (°C)": round(r["Cell_Temp [C]"],2)})
    if c3.button("Clear saved", use_container_width=True):
        st.session_state.scenarios = []
    if st.session_state.scenarios:
        saved = pd.DataFrame(st.session_state.scenarios)
        f = go.Figure(go.Bar(x=saved["Runtime (min)"], y=[f"{i+1}. {n}" for i,n in enumerate(saved.Scenario)], orientation="h", marker_color="#35658c", text=saved["Runtime (min)"], textposition="auto"))
        f.update_layout(xaxis_title="Predicted runtime (min)")
        st.plotly_chart(chart_style(f), width="stretch")
        st.dataframe(saved, hide_index=True, width="stretch")
        st.download_button("Download comparison CSV", saved.to_csv(index=False), "amr-scenarios.csv", "text/csv")
    else:
        st.info("No saved scenarios yet. Adjust the mission inputs, name the plan, and select Save current.")

with data_tab:
    st.subheader("Built on a physics-informed simulation")
    st.write("The supplied dataset contains 8,000 synthetic operating points. A StandardScaler → weighted XGBoost / LightGBM / CatBoost pipeline estimates remaining runtime from 15 inputs. Cell temperature, voltage and current in this dashboard come from the original physics equations; they are not separate model predictions or live sensor readings.")
    df = load_data()
    a,b,c = st.columns(3)
    a.metric("Synthetic samples", f"{len(df):,}")
    b.metric("Model inputs", len(model.feature_names_in_))
    c.metric("Nominal battery", "960 Wh")
    report = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(ROOT / "test_predictions.csv")
    mae, rmse, r2 = st.columns(3)
    mae.metric("Held-out MAE", f"{report['mae_minutes']:.3f} min")
    rmse.metric("Held-out RMSE", f"{report['rmse_minutes']:.3f} min")
    r2.metric("Held-out R²", f"{report['r2']:.5f}")
    st.caption(f"{report['train_rows']:,} training / {report['test_rows']:,} test rows · random seed {report['random_seed']} · synthetic data. Error metrics are not confidence intervals.")
    scatter_col, residual_col = st.columns(2)
    with scatter_col:
        st.markdown("**Held-out predictions**")
        fig = go.Figure(go.Scattergl(x=scores.actual_minutes, y=scores.predicted_minutes,
            mode="markers", marker=dict(size=4, opacity=.35, color="#35658c"), name="Test samples"))
        bounds = [0, max(scores.actual_minutes.max(), scores.predicted_minutes.max())]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", line=dict(color="#d38c40", dash="dot"), name="Perfect prediction"))
        fig.update_layout(xaxis_title="Actual runtime (min)", yaxis_title="Predicted runtime (min)")
        st.plotly_chart(chart_style(fig), width="stretch")
    with residual_col:
        st.markdown("**Residual distribution**")
        fig = go.Figure(go.Histogram(x=scores.predicted_minutes-scores.actual_minutes,
            nbinsx=45, marker_color="#35658c", name="Residuals"))
        fig.update_layout(xaxis_title="Predicted − actual (min)", yaxis_title="Test samples")
        st.plotly_chart(chart_style(fig), width="stretch")
    st.download_button("Download held-out predictions", scores.to_csv(index=False), "hybrid-test-predictions.csv", "text/csv")
    surface = st.multiselect("Dataset surfaces", df.Terrain_Surface.unique(), default=list(df.Terrain_Surface.unique()))
    filtered = df[df.Terrain_Surface.isin(surface)]
    distribution, importance = st.columns(2)
    with distribution:
        st.markdown("**Runtime distribution**")
        histogram = go.Figure(go.Histogram(x=filtered["Remaining_Runtime [min]"], nbinsx=35,
            marker_color="#35658c", name="Synthetic samples"))
        histogram.update_layout(xaxis_title="Remaining runtime (min)", yaxis_title="Samples", bargap=.08)
        st.plotly_chart(chart_style(histogram, 280), width="stretch")
    with importance:
        st.markdown("**What the fitted model uses**")
        selected_model = st.selectbox("Inspect a member", ["xgb", "lgbm", "cat"], format_func=lambda name: {"xgb":"XGBoost", "lgbm":"LightGBM", "cat":"CatBoost"}[name])
        raw = model.named_steps["ensemble"].named_estimators_[selected_model].feature_importances_
        normalized = raw / max(float(np.sum(raw)), 1e-12)
        weights = pd.Series(normalized, index=model.feature_names_in_).nlargest(6).sort_values()
        bars = go.Figure(go.Bar(x=weights.values, y=weights.index, orientation="h", marker_color="#779d98"))
        bars.update_layout(xaxis_title="Normalized feature importance")
        st.plotly_chart(chart_style(bars, 280), width="stretch")
        st.caption("Normalized within the selected member. Importance definitions differ by library; do not compare magnitudes across models. These are not causal effects.")
    st.dataframe(filtered, hide_index=True, height=320, width="stretch")
    st.caption(f"{len(filtered):,} rows selected")
    st.download_button("Download filtered telemetry", filtered.to_csv(index=False), "amr-telemetry.csv", "text/csv")
    with st.expander("Inspect the current model input"):
        st.dataframe(pd.DataFrame([r]).T.rename(columns={0:"Value"}), width="stretch")
    st.markdown("**Assumptions:** constant speed and load; a 10% discharge floor; no stop/start duty cycle, route geometry, charging losses or real BMS feed. Range is runtime × commanded speed. The model’s synthetic targets were clipped to 1–300 minutes.")

st.download_button("Export current mission", pd.DataFrame([{**asdict(mission), **r, "Planned duration [min]": duration, "Buffer [%]": reserve}]).to_csv(index=False), "amr-mission.csv", "text/csv")
st.markdown('<div class="foot">AMR / HYBRID STUDIO &nbsp; · &nbsp; 24 V LiFePO₄ platform &nbsp; · &nbsp; Simulated conditions / local inference</div>', unsafe_allow_html=True)

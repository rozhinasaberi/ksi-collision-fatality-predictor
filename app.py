import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KSI Fatality Predictor",
    page_icon="🚦",
    layout="centered",
)

# ── feature options (same as Flask app) ──────────────────────────────────────
FEATURE_OPTIONS = {
    "ROAD_CLASS": ["Major Arterial", "Minor Arterial", "Collector", "Local",
                   "Expressway", "Laneway", "Other"],
    "DISTRICT":   ["Toronto and East York", "North York", "Scarborough", "Etobicoke York"],
    "ACCLOC":     ["At Intersection", "Intersection Related", "Mid-Block",
                   "At/Near Private Drive", "Overpass or Bridge", "Trail",
                   "Underpass or Tunnel", "Other"],
    "TRAFFCTL":   ["Traffic Signal", "Stop Sign", "No Control", "Pedestrian Crossover",
                   "Traffic Controller", "Yield Sign", "School Guard",
                   "Police Control", "Other"],
    "VISIBILITY": ["Clear", "Rain", "Snow", "Fog, Mist, Smoke, Dust",
                   "Freezing Rain", "Strong wind", "Drifting Snow", "Other"],
    "LIGHT":      ["Daylight", "Dark", "Dusk", "Dawn", "Daylight, artificial",
                   "Dark, artificial", "Dusk, artificial", "Dawn, artificial", "Other"],
    "RDSFCOND":   ["Dry", "Wet", "Loose Snow", "Packed Snow", "Ice", "Slush",
                   "Loose Sand or Gravel", "Spilled liquid", "Other"],
    "IMPACTYPE":  ["Pedestrian Collisions", "Cyclist Collisions", "Rear End", "Angle",
                   "Sideswipe", "SMV Other", "Turning Movement", "Head On",
                   "SMV Unattended Vehicle", "Approaching", "Other"],
}

FLAG_COLUMNS = [
    "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
    "PASSENGER", "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY",
]

MODEL_REGISTRY = {
    "ksi_svm":           "Support Vector Machine (Best Model)",
    "ksi_decision_tree": "Decision Tree",
    "ksi_random_forest": "Random Forest",
}

PROJECT_DIR = Path(__file__).resolve().parent


# ── model loader (cached so it only loads once) ───────────────────────────────
@st.cache_resource
def load_model(filename: str):
    path = PROJECT_DIR / filename
    if not path.exists():
        st.error(f"Model file `{filename}` not found. Make sure it's in the same folder as app.py.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


# ── header ────────────────────────────────────────────────────────────────────
st.title("🚦 KSI Fatality Predictor")
st.markdown(
    "Predict whether a Toronto traffic collision is **Fatal** or **Non-Fatal** "
    "based on road, environmental, and involvement factors.  \n"
    "*Dataset: Toronto Police Service KSI (2006–2023)*"
)
st.divider()

# ── model selector ────────────────────────────────────────────────────────────
model_key = st.selectbox(
    "Select Model",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda k: MODEL_REGISTRY[k],
)
model = load_model(f"{model_key}.pkl")

st.divider()

# ── input form ────────────────────────────────────────────────────────────────
st.subheader("Collision Details")

col1, col2 = st.columns(2)

with col1:
    road_class  = st.selectbox("Road Class",        FEATURE_OPTIONS["ROAD_CLASS"])
    district    = st.selectbox("District",          FEATURE_OPTIONS["DISTRICT"])
    accloc      = st.selectbox("Accident Location", FEATURE_OPTIONS["ACCLOC"])
    traffctl    = st.selectbox("Traffic Control",   FEATURE_OPTIONS["TRAFFCTL"])

with col2:
    visibility  = st.selectbox("Visibility",        FEATURE_OPTIONS["VISIBILITY"])
    light       = st.selectbox("Light Condition",   FEATURE_OPTIONS["LIGHT"])
    rdsfcond    = st.selectbox("Road Surface",      FEATURE_OPTIONS["RDSFCOND"])
    impactype   = st.selectbox("Impact Type",       FEATURE_OPTIONS["IMPACTYPE"])

hour = st.slider("Hour of Day (24h)", min_value=0, max_value=23, value=14)

st.subheader("Involvement Flags")
flag_cols = st.columns(4)
flag_values = {}
for i, flag in enumerate(FLAG_COLUMNS):
    label = flag.replace("AG_DRIV", "Aggressive Driving").replace("REDLIGHT", "Ran Red Light").title()
    flag_values[flag] = flag_cols[i % 4].checkbox(label, value=False)

st.divider()

# ── prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Outcome", type="primary", use_container_width=True):
    input_dict = {col: [val] for col, val in zip(
        FEATURE_OPTIONS.keys(),
        [road_class, district, accloc, traffctl, visibility, light, rdsfcond, impactype]
    )}
    input_dict["HOUR"] = [hour]
    for flag in FLAG_COLUMNS:
        input_dict[flag] = [1 if flag_values[flag] else 0]

    input_df = pd.DataFrame(input_dict)

    try:
        prediction   = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        prob_fatal    = round(float(probabilities[1]) * 100, 1)
        prob_nonfatal = round(float(probabilities[0]) * 100, 1)
        confidence    = round(float(max(probabilities)) * 100, 1)

        if prediction == 1:
            st.error(f"### 🔴 Prediction: FATAL")
        else:
            st.success(f"### 🟢 Prediction: NON-FATAL")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fatal Probability",     f"{prob_fatal}%")
        c2.metric("Non-Fatal Probability", f"{prob_nonfatal}%")
        c3.metric("Model Confidence",      f"{confidence}%")

        st.caption(f"Model used: **{MODEL_REGISTRY[model_key]}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("COMP247 · Supervised Learning · Rojina Saberi (301533334) · Elijah Robinson (301194323)")

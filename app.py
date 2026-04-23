from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

# flask app starts here
app = Flask(__name__)

# just getting folder path stuff so app can find files no matter where we run it from
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# dropdown values for the frontend form
# we send these to react so user can just pick values instead of typing everything
FEATURE_OPTIONS = {
    "ROAD_CLASS": [
        "Major Arterial",
        "Minor Arterial",
        "Collector",
        "Local",
        "Expressway",
        "Laneway",
        "Other",
    ],
    "DISTRICT": [
        "Toronto and East York",
        "North York",
        "Scarborough",
        "Etobicoke York",
    ],
    "ACCLOC": [
        "At Intersection",
        "Intersection Related",
        "Mid-Block",
        "At/Near Private Drive",
        "Overpass or Bridge",
        "Trail",
        "Underpass or Tunnel",
        "Other",
    ],
    "TRAFFCTL": [
        "Traffic Signal",
        "Stop Sign",
        "No Control",
        "Pedestrian Crossover",
        "Traffic Controller",
        "Yield Sign",
        "School Guard",
        "Police Control",
        "Other",
    ],
    "VISIBILITY": [
        "Clear",
        "Rain",
        "Snow",
        "Fog, Mist, Smoke, Dust",
        "Freezing Rain",
        "Strong wind",
        "Drifting Snow",
        "Other",
    ],
    "LIGHT": [
        "Daylight",
        "Dark",
        "Dusk",
        "Dawn",
        "Daylight, artificial",
        "Dark, artificial",
        "Dusk, artificial",
        "Dawn, artificial",
        "Other",
    ],
    "RDSFCOND": [
        "Dry",
        "Wet",
        "Loose Snow",
        "Packed Snow",
        "Ice",
        "Slush",
        "Loose Sand or Gravel",
        "Spilled liquid",
        "Other",
    ],
    "IMPACTYPE": [
        "Pedestrian Collisions",
        "Cyclist Collisions",
        "Rear End",
        "Angle",
        "Sideswipe",
        "SMV Other",
        "Turning Movement",
        "Head On",
        "SMV Unattended Vehicle",
        "Approaching",
        "Other",
    ],
}

# these are the yes/no type columns from the form
# backend turns them into 1 or 0 before prediction
FLAG_COLUMNS = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "PASSENGER",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

# deployed models for our demo
# object starts as none then loads when needed
MODEL_REGISTRY = {
    "decision_tree": {
        "label": "Rojina Saberi - Decision Tree",
        "file": "ksi_decision_tree.pkl",
        "object": None,
    },
    "random_forest": {
        "label": "Elijah Robinson - Random Forest",
        "file": "ksi_random_forest.pkl",
        "object": None,
    },
    "svm": {
        "label": "Rojina Saberi - Support Vector Machine",
        "file": "ksi_svm.pkl",
        "object": None,
    },
}


def get_default_model_key() -> str:
    # opening model when page first loads
    return "svm"


# current selected model gets stored here
active_model_key = get_default_model_key()


def load_model(key: str):
    # this loads the pkl file only once
    # after that it keeps it in memory so switching is faster
    entry = MODEL_REGISTRY[key]
    if entry["object"] is None:
        model_path = PROJECT_DIR / entry["file"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{entry['file']}' not found. Run script.py first to generate it."
            )
        with open(model_path, "rb") as model_file:
            entry["object"] = pickle.load(model_file)
    return entry["object"]


def available_models() -> dict[str, str]:
    # only return models that actually exist in the folder
    return {
        key: value["label"]
        for key, value in MODEL_REGISTRY.items()
        if (PROJECT_DIR / value["file"]).exists()
    }


def build_input_frame(payload: dict[str, object]) -> pd.DataFrame:
    # make the incoming json look like one row of dataframe
    # this is what sklearn pipeline expects for predict
    input_dict = {column: [payload.get(column)] for column in FEATURE_OPTIONS}

    # hour comes from number input
    raw_hour = payload.get("HOUR", 12)
    hour = int(raw_hour)
    if not 0 <= hour <= 23:
        raise ValueError("HOUR must be between 0 and 23.")
    input_dict["HOUR"] = [hour]

    # checkbox true/false -> 1/0
    for flag in FLAG_COLUMNS:
        input_dict[flag] = [1 if payload.get(flag) else 0]

    return pd.DataFrame(input_dict)


@app.route("/")
def index():
    # this is the first route when browser opens
    # react gets all starting data from here
    initial_config = {
        "apiBase": "",
        "options": FEATURE_OPTIONS,
        "flagColumns": FLAG_COLUMNS,
        "modelRegistry": available_models(),
        "activeModel": active_model_key,
    }
    return render_template(
        "index.html",
        initial_config=initial_config,
    )


@app.route("/set_model", methods=["POST"])
def set_model():
    # changes which model is active on the page
    # frontend calls this when user clicks other model
    global active_model_key

    payload = request.get_json(silent=True) or {}
    model_key = payload.get("model")

    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": "Unknown model key."}), 400

    try:
        load_model(model_key)
        active_model_key = model_key
        return jsonify(
            {
                "success": True,
                "active_model": model_key,
                "label": MODEL_REGISTRY[model_key]["label"],
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/predict", methods=["POST"])
def predict():
    # main inference route
    # gets form values -> builds dataframe -> runs selected model
    payload = request.get_json(silent=True) or {}
    try:
        model = load_model(active_model_key)
        input_df = build_input_frame(payload)
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
    except Exception as exc:
        # if something breaks we send error back to frontend
        return jsonify({"error": str(exc)}), 400

    # returning both label and probs so frontend can show nicer result card
    return jsonify(
        {
            "prediction": prediction,
            "label": "Fatal" if prediction == 1 else "Non-Fatal",
            "prob_fatal": round(float(probabilities[1]), 4),
            "prob_non_fatal": round(float(probabilities[0]), 4),
            "confidence_pct": round(float(max(probabilities)) * 100, 1),
            "model_used": MODEL_REGISTRY[active_model_key]["label"],
        }
    )


@app.route("/health")
def health():
    # simple check route just to see if api is alive
    return jsonify(
        {
            "status": "ok",
            "active_model": active_model_key,
            "available_models": available_models(),
            "loaded_models": {
                key: value["object"] is not None for key, value in MODEL_REGISTRY.items()
            },
        }
    )


if __name__ == "__main__":
    # local run
    # for class demo we run this file, not script.py
    app.run(debug=True, host="0.0.0.0", port=5000)

from __future__ import annotations

import json
import math
import pickle
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# hiding warnings so output stays cleaner in terminal
warnings.filterwarnings("ignore")


# file paths used through the project
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "TOTAL_KSI.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = PROJECT_DIR / "reports"

# yes/no style columns from the dataset
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

# main categorical features we kept for modelling
CATEGORICAL_FEATURES = [
    "ROAD_CLASS",
    "DISTRICT",
    "ACCLOC",
    "TRAFFCTL",
    "VISIBILITY",
    "LIGHT",
    "RDSFCOND",
    "IMPACTYPE",
]

# hour + binary flags
NUMERIC_FEATURES = ["HOUR", *FLAG_COLUMNS]


def ensure_output_dirs() -> None:
    # make output folders if they dont exist yet
    for path in (ARTIFACTS_DIR, PLOTS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_one_hot_encoder() -> OneHotEncoder:
    # this try/except is just because sklearn versions can be slightly different
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    # full preprocessing pipeline used before model training
    # numeric -> impute + scale
    # categorical -> impute + one hot encode
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", build_one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def svg_document(width: int, height: int, elements: list[str]) -> str:
    # helper to build svg text manually for charts
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        + "".join(elements)
        + "</svg>"
    )


def save_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    # save svg chart file
    path.write_text(svg_document(width, height, elements))


def write_bar_chart_svg(
    path: Path,
    title: str,
    categories: list[str],
    values: list[float],
    colors: list[str],
    horizontal: bool = False,
) -> None:
    width, height = (920, 520) if horizontal else (860, 520)
    margin_left = 220 if horizontal else 70
    margin_right = 40
    margin_top = 60
    margin_bottom = 110 if not horizontal else 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)

    elements = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" '
        f'font-family="Arial" fill="#1f2937">{escape(title)}</text>',
    ]

    if horizontal:
        bar_height = plot_height / max(len(categories), 1) * 0.65
        gap = plot_height / max(len(categories), 1)
        for index, (category, value, color) in enumerate(zip(categories, values, colors)):
            y = margin_top + index * gap + (gap - bar_height) / 2
            bar_width = 0 if max_value == 0 else (value / max_value) * plot_width
            elements.extend(
                [
                    f'<text x="{margin_left - 10}" y="{y + bar_height / 2 + 4}" text-anchor="end" '
                    f'font-size="12" font-family="Arial" fill="#374151">{escape(str(category))}</text>',
                    f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" '
                    f'fill="{color}" rx="4" ry="4"/>',
                    f'<text x="{margin_left + bar_width + 8}" y="{y + bar_height / 2 + 4}" '
                    f'font-size="12" font-family="Arial" fill="#111827">{value}</text>',
                ]
            )
    else:
        bar_width = plot_width / max(len(categories), 1) * 0.65
        gap = plot_width / max(len(categories), 1)
        baseline = margin_top + plot_height
        elements.append(
            f'<line x1="{margin_left}" y1="{baseline}" x2="{margin_left + plot_width}" y2="{baseline}" '
            'stroke="#9ca3af" stroke-width="1"/>'
        )
        for index, (category, value, color) in enumerate(zip(categories, values, colors)):
            x = margin_left + index * gap + (gap - bar_width) / 2
            bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
            y = baseline - bar_height
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                    f'fill="{color}" rx="4" ry="4"/>',
                    f'<text x="{x + bar_width / 2}" y="{baseline + 20}" text-anchor="middle" '
                    f'font-size="11" font-family="Arial" fill="#374151" transform="rotate(25 {x + bar_width / 2} {baseline + 20})">{escape(str(category))}</text>',
                    f'<text x="{x + bar_width / 2}" y="{y - 8}" text-anchor="middle" '
                    f'font-size="12" font-family="Arial" fill="#111827">{value}</text>',
                ]
            )

    save_svg(path, width, height, elements)


def write_grouped_hour_svg(path: Path, nonfatal_counts: list[int], fatal_counts: list[int]) -> None:
    width, height = 960, 520
    margin_left, margin_right, margin_top, margin_bottom = 65, 30, 60, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(nonfatal_counts), max(fatal_counts), 1)
    slot_width = plot_width / 24
    bar_width = slot_width * 0.32
    baseline = margin_top + plot_height

    elements = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="480" y="30" text-anchor="middle" font-size="20" font-family="Arial" fill="#1f2937">Hour Distribution by Outcome</text>',
        f'<line x1="{margin_left}" y1="{baseline}" x2="{margin_left + plot_width}" y2="{baseline}" stroke="#9ca3af" stroke-width="1"/>',
    ]

    for hour in range(24):
        x = margin_left + hour * slot_width
        nf_height = (nonfatal_counts[hour] / max_value) * plot_height
        fatal_height = (fatal_counts[hour] / max_value) * plot_height
        elements.extend(
            [
                f'<rect x="{x + slot_width * 0.12}" y="{baseline - nf_height}" width="{bar_width}" height="{nf_height}" fill="#2ecc71" rx="3" ry="3"/>',
                f'<rect x="{x + slot_width * 0.56}" y="{baseline - fatal_height}" width="{bar_width}" height="{fatal_height}" fill="#e74c3c" rx="3" ry="3"/>',
                f'<text x="{x + slot_width / 2}" y="{baseline + 20}" text-anchor="middle" font-size="10" font-family="Arial" fill="#374151">{hour}</text>',
            ]
        )

    elements.extend(
        [
            '<rect x="700" y="70" width="12" height="12" fill="#2ecc71" rx="2" ry="2"/>',
            '<text x="720" y="80" font-size="12" font-family="Arial" fill="#374151">Non-Fatal</text>',
            '<rect x="810" y="70" width="12" height="12" fill="#e74c3c" rx="2" ry="2"/>',
            '<text x="830" y="80" font-size="12" font-family="Arial" fill="#374151">Fatal</text>',
        ]
    )
    save_svg(path, width, height, elements)


def write_heatmap_svg(path: Path, matrix: pd.DataFrame, title: str) -> None:
    size = len(matrix.columns)
    cell = 42
    width = 220 + cell * size
    height = 220 + cell * size
    offset_x, offset_y = 140, 80
    elements = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-family="Arial" fill="#1f2937">{escape(title)}</text>',
    ]

    for row_index, row_name in enumerate(matrix.index):
        y = offset_y + row_index * cell
        elements.append(
            f'<text x="{offset_x - 8}" y="{y + 26}" text-anchor="end" font-size="10" font-family="Arial" fill="#374151">{escape(str(row_name))}</text>'
        )
        for col_index, col_name in enumerate(matrix.columns):
            x = offset_x + col_index * cell
            value = float(matrix.iloc[row_index, col_index])
            intensity = int(255 - abs(value) * 120)
            if value >= 0:
                fill = f"rgb(255,{intensity},{intensity})"
            else:
                fill = f"rgb({intensity},{intensity},255)"
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#e5e7eb"/>',
                    f'<text x="{x + cell / 2}" y="{y + 25}" text-anchor="middle" font-size="9" font-family="Arial" fill="#111827">{value:.2f}</text>',
                ]
            )

    for col_index, col_name in enumerate(matrix.columns):
        x = offset_x + col_index * cell + cell / 2
        elements.append(
            f'<text x="{x}" y="{offset_y - 10}" text-anchor="start" font-size="10" font-family="Arial" fill="#374151" transform="rotate(-35 {x} {offset_y - 10})">{escape(str(col_name))}</text>'
        )

    save_svg(path, width, height, elements)


def write_confusion_matrix_svg(path: Path, label: str, matrix: np.ndarray) -> None:
    width, height = 520, 420
    cell = 120
    start_x, start_y = 180, 110
    max_value = max(int(matrix.max()), 1)
    elements = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-family="Arial" fill="#1f2937">{escape(label)} Confusion Matrix</text>',
        '<text x="240" y="85" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">Predicted Non-Fatal</text>',
        '<text x="360" y="85" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">Predicted Fatal</text>',
        '<text x="120" y="180" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">Actual Non-Fatal</text>',
        '<text x="120" y="300" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">Actual Fatal</text>',
    ]

    for row_index in range(2):
        for col_index in range(2):
            value = int(matrix[row_index, col_index])
            shade = 255 - int((value / max_value) * 140)
            fill = f"rgb({shade},{shade},{255 if row_index == col_index else shade})"
            x = start_x + col_index * cell
            y = start_y + row_index * cell
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#9ca3af"/>',
                    f'<text x="{x + cell / 2}" y="{y + 68}" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">{value}</text>',
                ]
            )

    save_svg(path, width, height, elements)


def write_roc_svg(path: Path, roc_payload: list[dict[str, object]]) -> None:
    width, height = 780, 560
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 60, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_height
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

    elements = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="390" y="30" text-anchor="middle" font-size="20" font-family="Arial" fill="#1f2937">ROC Curve Comparison</text>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + plot_width}" y2="{baseline_y}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left}" y2="{margin_top}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + plot_width}" y2="{margin_top}" stroke="#d1d5db" stroke-dasharray="5,5"/>',
    ]

    for tick in range(6):
        fraction = tick / 5
        x = margin_left + fraction * plot_width
        y = baseline_y - fraction * plot_height
        elements.extend(
            [
                f'<text x="{x}" y="{baseline_y + 22}" text-anchor="middle" font-size="11" font-family="Arial" fill="#374151">{fraction:.1f}</text>',
                f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-size="11" font-family="Arial" fill="#374151">{fraction:.1f}</text>',
            ]
        )

    for index, entry in enumerate(roc_payload):
        points = []
        for fpr_value, tpr_value in zip(entry["fpr"], entry["tpr"]):
            x = margin_left + float(fpr_value) * plot_width
            y = baseline_y - float(tpr_value) * plot_height
            points.append(f"{x},{y}")
        color = colors[index % len(colors)]
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )
        legend_y = 90 + index * 20
        elements.extend(
            [
                f'<line x1="510" y1="{legend_y}" x2="535" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="545" y="{legend_y + 4}" font-size="12" font-family="Arial" fill="#374151">{escape(entry["label"])} (AUC={entry["roc_auc"]:.3f})</text>',
            ]
        )

    save_svg(path, width, height, elements)


def load_raw_dataset() -> pd.DataFrame:
    # first step of whole project
    # load csv then convert raw columns into something models can use
    print("Loading dataset...")
    raw_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["DATE"])
    print(f"Loaded {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

    # yes -> 1, everything else -> 0
    for column in FLAG_COLUMNS:
        raw_df[column] = raw_df[column].apply(
            lambda value: 1 if str(value).strip().lower() == "yes" else 0
        )

    # making hour feature from raw time column
    raw_df["TIME_NUM"] = pd.to_numeric(raw_df["TIME"], errors="coerce")
    raw_df["HOUR"] = (raw_df["TIME_NUM"] // 100).fillna(0).clip(lower=0, upper=23).astype(int)

    # target column for classification
    # 1 means fatal, 0 means not fatal
    raw_df["FATAL_FLAG"] = np.where(
        (raw_df["INJURY"].astype(str).str.upper() == "FATAL")
        | (raw_df["ACCLASS"].astype(str).str.upper() == "FATAL"),
        1,
        0,
    )
    return raw_df


def build_collision_level_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    # original data is person-level so same collision can appear many times
    # here we group to one row per collision
    print("Aggregating to collision level...")
    event_df = (
        raw_df.groupby("ACCNUM")
        .agg(
            {
                "FATAL_FLAG": "max",
                "TIME_NUM": "first",
                "ROAD_CLASS": "first",
                "DISTRICT": "first",
                "ACCLOC": "first",
                "TRAFFCTL": "first",
                "VISIBILITY": "first",
                "LIGHT": "first",
                "RDSFCOND": "first",
                "IMPACTYPE": "first",
                "PEDESTRIAN": "max",
                "CYCLIST": "max",
                "AUTOMOBILE": "max",
                "MOTORCYCLE": "max",
                "TRUCK": "max",
                "PASSENGER": "max",
                "SPEEDING": "max",
                "AG_DRIV": "max",
                "REDLIGHT": "max",
                "ALCOHOL": "max",
                "DISABILITY": "max",
            }
        )
        .reset_index()
    )
    event_df["HOUR"] = (event_df["TIME_NUM"] // 100).fillna(0).clip(lower=0, upper=23).astype(int)
    event_df = event_df.drop(columns=["ACCNUM", "TIME_NUM"])
    print(f"Collision dataset shape: {event_df.shape[0]:,} rows x {event_df.shape[1]} columns")
    return event_df


def save_dataset_summaries(raw_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    # save summary csv/json files for report and presentation
    raw_df.dtypes.astype(str).rename("dtype").to_csv(ARTIFACTS_DIR / "raw_column_types.csv")
    raw_df.isna().sum().sort_values(ascending=False).rename("missing_values").to_csv(
        ARTIFACTS_DIR / "raw_missing_values.csv"
    )
    event_df.isna().sum().sort_values(ascending=False).rename("missing_values").to_csv(
        ARTIFACTS_DIR / "event_missing_values.csv"
    )
    event_df[["FATAL_FLAG", *NUMERIC_FEATURES]].describe().T.to_csv(
        ARTIFACTS_DIR / "event_numeric_summary.csv"
    )
    correlation_df = event_df[["FATAL_FLAG", *NUMERIC_FEATURES]].corr(numeric_only=True)
    correlation_df.to_csv(ARTIFACTS_DIR / "event_correlations.csv")

    top_rows = []
    for column in CATEGORICAL_FEATURES:
        for value, count in event_df[column].fillna("Missing").value_counts().head(10).items():
            top_rows.append({"column": column, "value": value, "count": int(count)})
    pd.DataFrame(top_rows).to_csv(ARTIFACTS_DIR / "categorical_top_values.csv", index=False)

    overview = {
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "collision_rows": int(event_df.shape[0]),
        "collision_columns": int(event_df.shape[1]),
        "fatal_collisions": int(event_df["FATAL_FLAG"].sum()),
        "non_fatal_collisions": int((event_df["FATAL_FLAG"] == 0).sum()),
        "fatal_rate_pct": round(float(event_df["FATAL_FLAG"].mean() * 100), 2),
    }
    (ARTIFACTS_DIR / "dataset_overview.json").write_text(json.dumps(overview, indent=2))


def save_exploration_visuals(event_df: pd.DataFrame) -> None:
    # save some simple visuals we can use in report/ppt
    target_counts = event_df["FATAL_FLAG"].value_counts().sort_index()
    write_bar_chart_svg(
        PLOTS_DIR / "class_balance.svg",
        "Collision-Level Class Balance",
        ["Non-Fatal", "Fatal"],
        [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))],
        ["#2ecc71", "#e74c3c"],
    )

    nonfatal_counts = (
        event_df.loc[event_df["FATAL_FLAG"] == 0, "HOUR"].value_counts().reindex(range(24), fill_value=0)
    )
    fatal_counts = (
        event_df.loc[event_df["FATAL_FLAG"] == 1, "HOUR"].value_counts().reindex(range(24), fill_value=0)
    )
    write_grouped_hour_svg(
        PLOTS_DIR / "hour_distribution.svg",
        nonfatal_counts.tolist(),
        fatal_counts.tolist(),
    )

    impact_counts = event_df["IMPACTYPE"].fillna("Missing").value_counts().head(8).sort_values()
    write_bar_chart_svg(
        PLOTS_DIR / "impact_types.svg",
        "Top Impact Types",
        impact_counts.index.tolist(),
        impact_counts.tolist(),
        ["#4f8ef7"] * len(impact_counts),
        horizontal=True,
    )

    correlation_df = pd.read_csv(ARTIFACTS_DIR / "event_correlations.csv", index_col=0)
    write_heatmap_svg(PLOTS_DIR / "correlation_heatmap.svg", correlation_df, "Numeric Correlation Heatmap")


def build_model_specs() -> dict[str, dict[str, object]]:
    # all tested models are listed here
  
    return {
        "logistic_regression": {
            "label": "Logistic Regression",
            "search_type": "grid",
            "estimator": LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
            ),
            "params": {"model__C": [0.5, 1.0, 2.0]},
        },
        # rojina model for demo + deployment
        "decision_tree": {
            "label": "Decision Tree",
            "search_type": "grid",
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            ),
            "params": {
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 10, 20],
                "model__min_samples_leaf": [1, 5, 10],
                "model__criterion": ["gini", "entropy"],
            },
        },
        # elijah model for demo + deployment
        "random_forest": {
            "label": "Random Forest",
            "search_type": "random",
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": randint(120, 360),
                "model__max_depth": [5, 10, 15, 20, None],
                "model__min_samples_split": randint(2, 18),
                "model__min_samples_leaf": randint(1, 8),
                "model__max_features": ["sqrt", "log2"],
            },
            "n_iter": 12,
        },
        #extra model 
        
        "svm": {
            "label": "Support Vector Machine",
            "search_type": "grid",
            "estimator": SVC(
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
            "params": {
                "model__C": [0.5, 1.0, 2.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        },
        "neural_network": {
            "label": "Neural Network",
            "search_type": "grid",
            "estimator": MLPClassifier(
                random_state=42,
                max_iter=500,
                early_stopping=True,
            ),
            "params": {
                "model__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.01],
            },
        },
    }


def build_search(model_spec: dict[str, object]):
    # wraps preprocessing + model together
    # then adds either grid search or random search
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model_spec["estimator"]),
        ]
    )
    if model_spec["search_type"] == "random":
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_spec["params"],
            n_iter=int(model_spec.get("n_iter", 10)),
            cv=5,
            scoring="f1",
            n_jobs=1,
            random_state=42,
            verbose=1,
        )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec["params"],
        cv=5,
        scoring="f1",
        n_jobs=1,
        verbose=1,
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    # this is the main training loop for all models
    results = []
    roc_payload = []

    for model_key, model_spec in build_model_specs().items():
        label = str(model_spec["label"])
        print(f"\nTraining {label}...")
        search = build_search(model_spec)
        search.fit(X_train, y_train)

        # best tuned version of that model
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        matrix = confusion_matrix(y_test, y_pred)

        result = {
            "Model Key": model_key,
            "Model": label,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc, 4),
            "Best CV F1": round(float(search.best_score_), 4),
            "Best Params": json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(result)
        roc_payload.append({"label": label, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc})

        # save trained pipeline so app.py can load it later
        with open(PROJECT_DIR / f"ksi_{model_key}.pkl", "wb") as model_file:
            pickle.dump(best_model, model_file)

        # save confusion matrix visual for this model
        write_confusion_matrix_svg(PLOTS_DIR / f"confusion_matrix_{model_key}.svg", label, matrix)
        print(
            f"Saved {label} | F1={result['F1 Score']:.4f} | ROC-AUC={result['ROC-AUC']:.4f}"
        )

    write_roc_svg(PLOTS_DIR / "roc_curves.svg", roc_payload)
    results_df = pd.DataFrame(results).sort_values(
        by=["F1 Score", "ROC-AUC", "Recall"], ascending=False
    )
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    return results_df


def save_best_model_bundle(results_df: pd.DataFrame) -> None:
    # not used in final frontend now, but still nice to keep
    # stores whichever model had best score overall
    best_row = results_df.iloc[0]
    best_key = best_row["Model Key"]
    with open(PROJECT_DIR / f"ksi_{best_key}.pkl", "rb") as model_file:
        best_model = pickle.load(model_file)
    with open(PROJECT_DIR / "ksi_best_model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

    best_summary = {
        "best_model_key": best_key,
        "best_model_label": best_row["Model"],
        "selection_basis": "Highest F1 score, then ROC-AUC, then Recall",
        "metrics": {
            "accuracy": best_row["Accuracy"],
            "precision": best_row["Precision"],
            "recall": best_row["Recall"],
            "f1_score": best_row["F1 Score"],
            "roc_auc": best_row["ROC-AUC"],
        },
    }
    (ARTIFACTS_DIR / "best_model_summary.json").write_text(json.dumps(best_summary, indent=2))


def save_text_summary(results_df: pd.DataFrame) -> None:
    # quick text summary for report 
    overview = json.loads((ARTIFACTS_DIR / "dataset_overview.json").read_text())
    best_row = results_df.iloc[0]
    lines = [
        "COMP 247 KSI Fatality Predictor Summary",
        "=======================================",
        "",
        f"Raw dataset rows: {overview['raw_rows']:,}",
        f"Collision-level rows: {overview['collision_rows']:,}",
        f"Fatal collisions: {overview['fatal_collisions']:,}",
        f"Fatal collision rate: {overview['fatal_rate_pct']}%",
        "",
        "Feature set used:",
        f"- Categorical: {', '.join(CATEGORICAL_FEATURES)}",
        f"- Numeric/Binary: {', '.join(NUMERIC_FEATURES)}",
        "",
        "Preprocessing and modelling notes:",
        "- Converted all Yes/blank indicator columns to binary numeric flags.",
        "- Aggregated records by ACCNUM so each row represents one collision event.",
        "- Imputed missing numeric values with medians and categorical values with most-frequent values.",
        "- One-hot encoded categorical features and standardized numeric inputs.",
        "- Used stratified train/test splitting and class-weighted models where supported.",
        "",
        "Model ranking:",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"- {row['Model']}: Accuracy={row['Accuracy']}, Precision={row['Precision']}, Recall={row['Recall']}, F1={row['F1 Score']}, ROC-AUC={row['ROC-AUC']}"
        )
    lines.extend(
        [
            "",
            f"Recommended best model: {best_row['Model']}",
            f"Selection reason: best F1 score ({best_row['F1 Score']}) with ROC-AUC {best_row['ROC-AUC']}.",
        ]
    )
    (REPORTS_DIR / "analysis_summary.txt").write_text("\n".join(lines))


def main() -> None:
    # whole project flow starts here
    # create folders -> load data -> preprocess -> train -> save outputs
    ensure_output_dirs()
    raw_df = load_raw_dataset()
    event_df = build_collision_level_dataset(raw_df)
    save_dataset_summaries(raw_df, event_df)
    save_exploration_visuals(event_df)

    # x = features, y = target
    X = event_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = event_df["FATAL_FLAG"]

    # stratify so fatal/non-fatal balance stays similar in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_best_model_bundle(results_df)
    save_text_summary(results_df)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    # run full training script from here
    main()

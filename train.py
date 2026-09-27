from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "network_traffic.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

if not DATA.exists():
    import generate_dataset
    generate_dataset.main() if hasattr(generate_dataset, "main") else None

df = pd.read_csv(DATA)

target = "label"
X = df.drop(columns=[target])
y = df[target]

categorical = ["protocol", "service"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {}

models["Random Forest"] = RandomForestClassifier(
    n_estimators=250, max_depth=18, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)

try:
    from xgboost import XGBClassifier
    label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
    inv_map = {v: k for k, v in label_map.items()}
    y_train_xgb = y_train.map(label_map)
    models["XGBoost"] = XGBClassifier(
        n_estimators=250, max_depth=7, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
        num_class=len(label_map), eval_metric="mlogloss",
        random_state=42, n_jobs=4
    )
except Exception as e:
    print("XGBoost unavailable:", e)

try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=250, learning_rate=0.08, num_leaves=31,
        max_depth=-1, random_state=42, verbosity=-1
    )
except Exception as e:
    print("LightGBM unavailable:", e)

results = {}
best_name = None
best_score = -1
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if name == "XGBoost":
        pipe.fit(X_train, y_train_xgb)
        pred_num = pipe.predict(X_test).astype(int).ravel()
        pred = pd.Series(pred_num).map(inv_map).values
    else:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

    f1 = f1_score(y_test, pred, average="weighted")
    results[name] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1), 4),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "labels": sorted(y.unique().tolist())
    }

    if f1 > best_score:
        best_score = f1
        best_name = name
        best_pipeline = pipe

    print(f"{name}: accuracy={results[name]['accuracy']:.4f}, F1={f1:.4f}")

joblib.dump(best_pipeline, MODELS / "ids_model.joblib")
metadata = {
    "best_model": best_name,
    "features": X.columns.tolist(),
    "categorical_features": categorical,
    "numeric_features": numeric,
    "classes": sorted(y.unique().tolist()),
    "results": results
}
with open(MODELS / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Feature importance for the best tree model where available
try:
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        transformed_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        fi = pd.DataFrame({
            "feature": transformed_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)
        fi.to_csv(REPORTS / "feature_importance.csv", index=False)
except Exception:
    pass

print("\nTraining complete.")
print("Best model:", best_name)
print("Saved:", MODELS / "ids_model.joblib")

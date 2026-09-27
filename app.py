from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "ids_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "network_traffic.csv"

app = Flask(__name__)

FEATURES = [
    "duration", "src_bytes", "dst_bytes", "packets", "protocol", "service",
    "src_port", "dst_port", "failed_connections", "connection_rate",
    "unique_dst_ports", "bytes_per_packet", "syn_count", "rst_count",
    "http_requests"
]

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def stats():
    if not DATA_PATH.exists():
        return {}
    df = pd.read_csv(DATA_PATH)
    counts = df["label"].value_counts().to_dict()
    return {
        "total": int(len(df)),
        "attacks": int((df["label"] != "BENIGN").sum()),
        "benign": int((df["label"] == "BENIGN").sum()),
        "attack_types": counts
    }

@app.route("/")
def index():
    metadata = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return render_template("index.html", stats=stats(), metadata=metadata)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model not trained. Run python train.py first."}), 400

    data = request.get_json(silent=True) or request.form.to_dict()
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400

    row = {}
    numeric = set(FEATURES) - {"protocol", "service"}
    try:
        for f in FEATURES:
            row[f] = data[f] if f in {"protocol", "service"} else float(data[f])
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric feature contains an invalid value."}), 400

    X = pd.DataFrame([row])
    prediction = model.predict(X)[0]

    response = {"prediction": str(prediction)}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        response["confidence"] = {
            str(c): round(float(p), 4) for c, p in zip(classes, probabilities)
        }
    return jsonify(response)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

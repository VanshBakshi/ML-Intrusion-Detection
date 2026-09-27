from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).parent
model_path = ROOT / "models" / "ids_model.joblib"

if not model_path.exists():
    print("Model not found. Run: python train.py")
    raise SystemExit(1)

model = joblib.load(model_path)

sample = {
    "duration": 2.4,
    "src_bytes": 1200,
    "dst_bytes": 800,
    "packets": 15,
    "protocol": "tcp",
    "service": "http",
    "src_port": 44321,
    "dst_port": 80,
    "failed_connections": 0,
    "connection_rate": 5.2,
    "unique_dst_ports": 1,
    "bytes_per_packet": 133.3,
    "syn_count": 2,
    "rst_count": 0,
    "http_requests": 3
}

df = pd.DataFrame([sample])
prediction = model.predict(df)[0]

print("=" * 50)
print("ML INTRUSION DETECTION PREDICTION")
print("=" * 50)
print("Prediction:", prediction)

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(df)[0]
    classes = model.classes_
    print("\nConfidence:")
    for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
        print(f"  {cls:12} {prob:.2%}")

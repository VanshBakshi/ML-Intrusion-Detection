# ML Intrusion Detection System

An end-to-end Machine Learning Intrusion Detection System (IDS) built with Python.

## Features
- Synthetic network traffic dataset included
- Binary classification: BENIGN vs ATTACK
- Attack categories: DoS, PortScan, BruteForce, WebAttack, Botnet
- Data preprocessing and feature engineering
- Random Forest baseline
- XGBoost and LightGBM comparison when installed
- Automatic best-model selection using F1 score
- Confusion matrix and classification report
- Flask web dashboard
- Single-traffic prediction
- REST API endpoint
- Training history and saved model
- No external dataset is required to start

## Project Structure

ML_Intrusion_Detection/
├── app.py
├── train.py
├── generate_dataset.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   └── network_traffic.csv
├── models/
├── reports/
├── static/
│   └── style.css
└── templates/
    └── index.html

## Run in VS Code

Open the project folder in VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python app.py
```

Then open:
http://127.0.0.1:5000

If XGBoost or LightGBM installation fails on your Python version, the project automatically falls back to the Scikit-learn Random Forest model.

## Predict from Python

```powershell
python predict.py
```

## API

POST JSON to `/api/predict`:

```json
{
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
```

## Important

This is a defensive educational IDS project. The included data is synthetic and is intended for learning, demonstrations, testing and portfolio use. For production security monitoring, train and validate on properly collected, authorized network telemetry.

import joblib
import pandas as pd
import json
import os
from flask import Flask, request, jsonify, Blueprint
from pathlib import Path


ARTIFACTS_DIR = Path(
    os.getenv(
        "ARTIFACTS_DIR",
        "/home/aleksei/Documentos/python/Kibernum/Machine Learning/Trabajo en clase/M10/EF/artifacts",
    )
)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
MANIFEST_PATH = ARTIFACTS_DIR / "model_card.json"

model = joblib.load(MODEL_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

bp = Blueprint('bp', __name__)

@bp.get("/")
def home():
    return jsonify({"status": "OK"})


@bp.get("/model-info")
def model_info():
    return manifest


@bp.post("/predict")
def predict():
    data = request.get_json(force=True)
    data = pd.DataFrame(data)
    prediction = model.predict(data)
    data["prediction"] = prediction
    return jsonify({"ok": True, "results": data.to_dict(orient="records")})

def create_app():
    app = Flask("MI MODELO")
    app.register_blueprint(bp)    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port="5001")

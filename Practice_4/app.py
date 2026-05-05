from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("insurance_model.pkl")

app = FastAPI(title="Insurance Charges Prediction Service")


class InsuranceRequest(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/predict")
def predict(data: InsuranceRequest):
    if not MODEL_PATH.exists():
        return {
            "error": "Model file not found. Run train_model.py first."
        }

    bundle = joblib.load(MODEL_PATH)

    feature_columns = bundle["feature_columns"]
    feature_scaler = bundle["feature_scaler"]
    target_transformer = bundle["target_transformer"]
    model = bundle["model"]

    row = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region,
    }], columns=feature_columns)

    row_scaled = feature_scaler.transform(row)
    prediction_transformed = model.predict(row_scaled).reshape(-1, 1)
    prediction = target_transformer.inverse_transform(prediction_transformed)[0][0]

    return {
        "predicted_charges": round(float(prediction), 2),
        "input": data.model_dump()
    }
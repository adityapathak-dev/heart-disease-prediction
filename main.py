
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "heart_disease_rf_model.pkl")


class PatientData(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch: float
    exang: str
    oldpeak: float
    slope: str
    ca: float
    thal: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        patient_data = pd.DataFrame([patient.model_dump()])

        prediction = int(model.predict(patient_data)[0])
        probabilities = model.predict_proba(patient_data)[0]

        # Find the probability corresponding to class 1.
        class_index = list(model.classes_).index(1)
        probability = float(probabilities[class_index])

        return {
            "prediction": prediction,
            "result": (
                "Patient has heart disease."
                if prediction == 1
                else "Patient is normal."
            ),
            "model_probability": round(probability, 4),
            "disclaimer": (
                "Educational demonstration only; "
                "not a medical diagnosis."
            )
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
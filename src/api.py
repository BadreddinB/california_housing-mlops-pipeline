import joblib
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel


# ======================
# Load artifacts
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

app = FastAPI(title="California Housing Price Prediction API")

# ======================
# Input schema
# ======================
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    RoomsPerPerson: float
    BedroomsRatio: float
    PopulationDensity: float

# ======================
# Health check
# ======================
@app.get("/health")
def health():
    return {"status": "API is running"}

# ======================
# Prediction endpoint
# ======================
@app.post("/predict")
def predict(data: HouseFeatures):
    input_data = np.array([list(data.dict().values())])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return {"predicted_price": float(prediction[0])}
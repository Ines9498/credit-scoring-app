from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn

app = FastAPI(title="Credit Scoring API", description="API pour prédire si un client est bon ou mauvais payeur", version="1.0")

# Charger le modèle LightGBM entraîné
with open("lightgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Définir le format des données attendues
class ClientData(BaseModel):
    data: dict

@app.post("/predict")
def predict(client: ClientData):
    try:
        df = pd.DataFrame([client.data])
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1]
        return {
            "prediction": int(prediction),
            "proba_good_client": round(float(prediction_proba), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ ➕ Cette route garde le service actif sur Render
@app.get("/")
def read_root():
    return {"status": "API is running"}

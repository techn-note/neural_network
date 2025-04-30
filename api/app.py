from fastapi import FastAPI
import joblib
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Carrega artefatos
scaler = joblib.load("artifacts/scaler.pkl")
ohe = joblib.load("artifacts/ohe.pkl")
model = tf.keras.models.load_model("artifacts/modelo_tilapia.h5")

class PredictionRequest(BaseModel):
    temperatura: float
    ph: float
    tds: float
    volume: float
    fase: str     # Ex: "alevino"
    hora: int     # 0-23

@app.post("/predict")
def predict(request: PredictionRequest):
    # Codificação da fase
    fase_encoded = ohe.transform([[request.fase]]).toarray()
    
    # Hora senoidal
    hora_sin = np.sin(2 * np.pi * request.hora / 24)
    hora_cos = np.cos(2 * np.pi * request.hora / 24)
    
    # Pré-processamento
    features = np.array([[request.temperatura, request.ph, request.tds, request.volume]])
    features_scaled = scaler.transform(features)
    
    # Concatenação final
    X = np.hstack([
        features_scaled,
        fase_encoded,
        np.array([[hora_sin, hora_cos]])
    ])
    
    # Predição
    prob = model.predict(X, verbose=0)[0][0]
    return {
        "probabilidade": float(prob),
        "status": "ALERTA" if prob > 0.5 else "OK"
    }

@app.get("/healthcheck")
def healthcheck():
    return {"status": "online"}
from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Carrega artefatos
scaler = joblib.load("/app/artifacts/scaler.pkl")
ohe = joblib.load("/app/artifacts/ohe.pkl")
model = tf.keras.models.load_model("/app/artifacts/modelo_tilapia.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recebe os dados da requisição
        data = request.get_json()
        temperatura = data["temperatura"]
        ph = data["ph"]
        tds = data["tds"]
        volume = data["volume"]
        fase = data["fase"]
        hora = data["hora"]

        # Pré-processamento
        features = np.array([[temperatura, ph, tds, volume]])
        features_scaled = scaler.transform(features)
        fase_encoded = ohe.transform([[fase]])
        sin_hora = np.sin(2 * np.pi * hora / 24)
        cos_hora = np.cos(2 * np.pi * hora / 24)

        # Combinação final
        X = np.hstack([features_scaled, fase_encoded, [[sin_hora, cos_hora]]])

        # Predição
        prob = model.predict(X, verbose=0)[0][0]
        return jsonify({
            "probabilidade_ideal": round(float(prob), 3),
            "status": "OK" if prob > 0.5 else "ALERTA"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "online"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
import shap
import os

app = Flask(__name__)

# Carrega artefatos
scaler = joblib.load("/app/artifacts/scaler.pkl")
ohe    = joblib.load("/app/artifacts/ohe.pkl")
model  = tf.keras.models.load_model("/app/artifacts/modelo_tilapia.h5")

# Background para SHAP
X_background = np.load('/app/artifacts/X.npy')
mask         = np.random.choice(X_background.shape[0], 100, replace=False)
background   = X_background[mask]
explainer    = shap.GradientExplainer(model, [background])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data        = request.get_json()
        temperatura = data["temperatura"]
        ph          = data["ph"]
        tds         = data["tds"]
        volume      = data["volume"]
        fase        = data["fase"]
        hora        = data["hora"]

        # Pré-processamento
        features        = np.array([[temperatura, ph, tds, volume]])
        features_scaled = scaler.transform(features)
        fase_encoded    = ohe.transform([[fase]])
        sin_hora        = np.sin(2 * np.pi * hora / 24)
        cos_hora        = np.cos(2 * np.pi * hora / 24)

        X_input = np.hstack([features_scaled, fase_encoded, [[sin_hora, cos_hora]]])

        # Predição multi-output
        probs = model.predict(X_input, verbose=0)[0]  # vetor de 4 valores

        # Convertendo valores para tipos serializáveis
        probs = [float(p) for p in probs]  # Converte para float padrão do Python

        # SHAP values para cada saída
        shap_vals = explainer.shap_values(X_input)
        shap_dict = {
            'temperatura': float(shap_vals[0][0][0]),
            'pH':          float(shap_vals[1][0][0]),
            'TDS':         float(shap_vals[2][0][0]),
            'volume':      float(shap_vals[3][0][0])
        }

        return jsonify({
            "probabilidades_ideais": {
                "temperatura": round(probs[0], 15),  # Mais precisão
                "pH":          round(probs[1], 15),
                "TDS":         round(probs[2], 15),
                "volume":      round(probs[3], 15)
            },
            "shap": shap_dict,
            "status": "OK"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "online"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

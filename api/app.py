from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
import shap

app = Flask(__name__)

# Carrega artefatos
scaler = joblib.load("/app/artifacts/scaler.pkl")
ohe    = joblib.load("/app/artifacts/ohe.pkl")
model  = tf.keras.models.load_model("/app/artifacts/modelo_tilapia.h5")

# Prepara o background usando X.npy
X_background = np.load('/app/artifacts/X.npy')                                    # :contentReference[oaicite:6]{index=6}
mask         = np.random.choice(X_background.shape[0], 100, replace=False)        # :contentReference[oaicite:7]{index=7}
background   = X_background[mask]
explainer    = shap.GradientExplainer(model, [background])                         # :contentReference[oaicite:8]{index=8}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recebe dados da requisição
        data        = request.get_json()
        temperatura = data["temperatura"]
        ph          = data["ph"]
        tds         = data["tds"]
        volume      = data["volume"]
        fase        = data["fase"]
        hora        = data["hora"]

        # Pré-processamento
        features        = np.array([[temperatura, ph, tds, volume]])
        features_scaled = scaler.transform(features)                                # :contentReference[oaicite:9]{index=9}
        fase_encoded    = ohe.transform([[fase]])                                   # :contentReference[oaicite:10]{index=10}
        sin_hora        = np.sin(2 * np.pi * hora / 24)
        cos_hora        = np.cos(2 * np.pi * hora / 24)

        X_input = np.hstack([features_scaled, fase_encoded, [[sin_hora, cos_hora]]])

        # Predição
        prob = float(model.predict(X_input, verbose=0)[0][0])

        # 7) Cálculo dos valores SHAP (ajustado para regressão com um único output)
        shap_values = explainer.shap_values(X_input)[0][0]                          
        
        feature_names = (
            ['temperatura','pH','TDS','volume'] +
            list(ohe.get_feature_names_out(['fase'])) +
            ['sin_hora','cos_hora']
        )
        shap_dict = {name: float(val) for name, val in zip(feature_names, shap_values)}

        # Resposta JSON
        return jsonify({
            "probabilidade_ideal": round(prob, 3),
            "status": "OK" if prob > 0.5 else "ALERTA",
            "shap": shap_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "online"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

from flask import Flask, request, jsonify
import joblib, tensorflow as tf, numpy as np, shap, json, os
app = Flask(__name__)

# Carrega artefatos
scaler = joblib.load('/app/artifacts/scaler.pkl')
model  = tf.keras.models.load_model('/app/artifacts/safira.h5')

# Carrega configuração de estágios para fuzzy
cfg_path = os.getenv('ESTAGIOS_CONFIG','ml/estagios_config.json')
ESTAGIOS = {e['estagio']: e for e in json.load(open(cfg_path))}

# SHAP background (apenas features numéricas)
Xb = np.load('/app/artifacts/X.npy')
bg = Xb[np.random.choice(Xb.shape[0], 100, replace=False)]
explainer = shap.GradientExplainer(model, [bg])


def fuzzy_score(x, low_tol, low_ideal, high_ideal, high_tol):
    if x < low_tol or x > high_tol:
        return 0.0
    if low_ideal <= x <= high_ideal:
        return 1.0
    if x < low_ideal:
        return (x - low_tol) / (low_ideal - low_tol)
    return (high_tol - x) / (high_tol - high_ideal)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        temp = data['temperatura']
        ph   = data['ph']
        tds  = data['tds']

        # Pré-processamento simples: apenas scale numérico
        features = np.array([[temp, ph, tds]])
        X_input = scaler.transform(features)

        # Adiciona a fase ao vetor de entrada
        fase = data['fase']
        fase_encoded = np.zeros(len(ESTAGIOS))  # Vetor one-hot para fases
        if fase in ESTAGIOS:
            fase_index = list(ESTAGIOS.keys()).index(fase)
            fase_encoded[fase_index] = 1
        else:
            raise ValueError(f"Fase desconhecida: {fase}")

        # Concatena as features numéricas com a fase codificada
        X_input = np.hstack([X_input, fase_encoded.reshape(1, -1)])

        # Predição Softmax
        probs = model.predict(X_input, verbose=0)[0]
        classes = ['Crise', 'Anormal', 'Normal']
        prob_dict = {c: round(float(p), 3) for c, p in zip(classes, probs)}
        pred_class = classes[np.argmax(probs)]

        # Cálculo de fuzzy scores para cada parâmetro
        stage = data.get('fase')  # opcional para fuzzy scores
        scores = None
        if stage and stage in ESTAGIOS:
            ranges = ESTAGIOS[stage]['ranges']
            scores = {
                'temperatura': round(fuzzy_score(temp, ranges['temperatura']['min'], ranges['temperatura'].get('ideal_min', ranges['temperatura']['min']), ranges['temperatura'].get('ideal_max', ranges['temperatura']['max']), ranges['temperatura']['max']) * 100, 2),
                'pH':          round(fuzzy_score(ph, ranges['ph']['min'], ranges['ph'].get('ideal_min', ranges['ph']['min']), ranges['ph'].get('ideal_max', ranges['ph']['max']), ranges['ph']['max']) * 100, 2),
                'tds':         round(fuzzy_score(tds, ranges['tds']['min'], ranges['tds'].get('ideal_min', ranges['tds']['min']), ranges['tds'].get('ideal_max', ranges['tds']['max']), ranges['tds']['max']) * 100, 2)
            }

        # SHAP values
        shap_vals = explainer.shap_values(X_input)[0][0]
        feature_names = ['temperatura', 'ph', 'tds']
        shap_dict = {n: float(v) for n, v in zip(feature_names, shap_vals)}

        response = {
            'classificacao': pred_class,
            'probabilidades': prob_dict,
            'shap': shap_dict
        }
        if scores is not None:
            response['scores_percent'] = scores

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def health():
    return jsonify({'status': 'online'})

if __name__ == '__main__':
    app.run('0.0.0.0', 8000)
import numpy as np
import joblib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client.piscicultura

def load_data():
    leituras = list(db.leituras.find())
    X_num, fases, horas, Ys = [], [], [], []

    for doc in leituras:
        e = db.estagios.find_one({'_id': doc['faseId']})
        if not e:
            continue
        # features numéricas
        temp = doc['temperatura']; ph = doc['pH']
        tds = doc['TDS']; vol = doc['volume']
        hora = doc['timestamp'].hour

        # calcula score contínuo fuzzy
        score = []
        for param in ['temperatura','pH','TDS','volume']:
            low_i, high_i = e['ranges'][param]['ideal']
            low_t, high_t = e['ranges'][param]['tol']
            val = doc[param]
            if val < low_t or val > high_t:
                s = 0.0
            elif low_i <= val <= high_i:
                s = 1.0
            elif val < low_i:
                s = (val - low_t) / (low_i - low_t)
            else:
                s = (high_t - val) / (high_t - high_i)
            score.append(np.clip(s, 0, 1))

        # y final como média dos scores
        Ys.append(float(np.mean(score)))
        X_num.append([temp, ph, tds, vol])
        fases.append([doc['faseId']])
        horas.append(hora)

    return np.array(X_num), np.array(fases), np.array(horas), np.array(Ys)


def preprocess():
    X_num, fases, horas, y = load_data()

    # normalização numérica
    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # codifica estágio
    ohe = OneHotEncoder(sparse_output=False)
    X_fase = ohe.fit_transform(fases)

    # hora cíclica
    sin_h = np.sin(2 * np.pi * horas / 24)
    cos_h = np.cos(2 * np.pi * horas / 24)

    X = np.hstack([X_num_scaled, X_fase, sin_h.reshape(-1,1), cos_h.reshape(-1,1)])

    # salva artefatos
    os.makedirs('/app/artifacts', exist_ok=True)
    joblib.dump(scaler, '/app/artifacts/scaler.pkl')
    joblib.dump(ohe, '/app/artifacts/ohe.pkl')
    np.save('/app/artifacts/X.npy', X)
    np.save('/app/artifacts/y.npy', y)

    print("Pré-processamento concluído! X shape:", X.shape)
    return X, y

if __name__ == '__main__':
    preprocess()
import os
import json
import numpy as np
import joblib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf


client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client.piscicultura
with open(os.getenv('ESTAGIOS_CONFIG', 'ml/estagios_config.json'), 'r') as f:
    ESTAGIOS = json.load(f)

def load_data():
    docs = list(db.leituras.find())
    X_num, fases, Ys = [], [], []
    for doc in docs:
        X_num.append([
            doc.get('temperatura', 0.0),
            doc.get('ph', 0.0),
            doc.get('tds', 1.0)  # garante mínimo
        ])
        fases.append(doc.get('faseId'))
        Ys.append(doc.get('class'))
    return np.array(X_num), np.array(fases), np.array(Ys)


def preprocess():
    X_num, fases, y_labels = load_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_num)

    ohe = OneHotEncoder(sparse_output=False)
    X_fase = ohe.fit_transform(fases.reshape(-1,1))

    X = np.hstack([X_scaled, X_fase])

    le = LabelEncoder()
    y_int = le.fit_transform(y_labels)
    y = tf.keras.utils.to_categorical(y_int, num_classes=3)

    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    joblib.dump(ohe, 'artifacts/ohe.pkl')
    joblib.dump(le, 'artifacts/le.pkl')
    np.save('artifacts/X.npy', X)
    np.save('artifacts/y.npy', y)

    print(f"Pré-processamento concluído: X={X.shape}, y={y.shape}")
    return X, y

if __name__ == '__main__':
    preprocess()
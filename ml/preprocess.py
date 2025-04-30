import numpy as np
import joblib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

client = MongoClient(os.getenv("MONGO_URI"))
db = client.piscicultura

def load_data():
    leituras = list(db.leituras.find())
    X = []
    y = []
    
    for doc in leituras:
        try:
            # Features
            temp = doc['temperatura']
            ph = doc['pH']
            tds = doc['TDS']
            vol = doc['volume']
            fase = doc['faseId']
            hora = doc['timestamp'].hour
            
            # Busca os dados do estágio
            fase_info = db.estagios.find_one({"_id": fase})
            if not fase_info:
                print(f"Estágio {fase} não encontrado! Pulando leitura...")
                continue

            # Verifica campos obrigatórios
            if not all(key in fase_info for key in ['temp_ideal', 'pH_ideal', 'tolerancias']):
                print(f"Estágio {fase} com campos incompletos! Pulando...")
                continue

            # Label (0 = OK, 1 = ALERTA)
            alerta = False
            for param in ['temp', 'pH']:  # Campos corrigidos para match com data_generation.py
                val = doc['temperatura' if param == 'temp' else 'pH']
                ideal_low, ideal_high = fase_info[f"{param}_ideal"]
                tol_low, tol_high = fase_info['tolerancias'][param]
                
                if not (tol_low <= val <= tol_high):
                    alerta = True
            
            y.append(1 if alerta else 0)
            X.append([temp, ph, tds, vol, fase, hora])
        
        except KeyError as e:
            print(f"Documento inválido: campo {e} não encontrado. Pulando...")
            continue
    
    return np.array(X), np.array(y)
def preprocess():
    X, y = load_data()
    
    # Separa features
    num_features = X[:, :4].astype(float)
    fase = X[:, 4].reshape(-1, 1)
    hora = X[:, 5].astype(int)
    
    # Normalização
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(num_features)
    
    # Codificação
    ohe = OneHotEncoder(sparse_output=False)
    fase_encoded = ohe.fit_transform(fase)
    
    # Hora senoidal
    sin_hora = np.sin(2 * np.pi * hora / 24)
    cos_hora = np.cos(2 * np.pi * hora / 24)
    
    # Combinação final
    X_processed = np.hstack([
        num_scaled,
        fase_encoded,
        sin_hora.reshape(-1, 1),
        cos_hora.reshape(-1, 1)
    ])
    
    # Salva artefatos
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(ohe, 'ohe.pkl')
    np.save('X.npy', X_processed)
    np.save('y.npy', y)
    
    print("Pré-processamento concluído!")
    return X_processed, y

if __name__ == "__main__":
    preprocess()
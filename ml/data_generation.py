import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta

# Conexão com MongoDB LOCAL
client = MongoClient("mongodb://mongo:27018/")
db = client.piscicultura


def populate_estagios():
    estagios = db.estagios
    if estagios.estimated_document_count() == 0:
        estagios.insert_many([
            {
                "_id": "alevino",
                "nome": "Alevino",
                "temp_ideal": [28, 30],
                "pH_ideal": [7.0, 8.0],
                "TDS_ideal": [600, 800],
                "volume_ideal": [1, 2],
                "tolerancias": {"temp": [25, 32], "pH": [6.5, 8.5]}
            },
            {
                "_id": "fry",
                "nome": "Fry",
                "temp_ideal": [27, 29],
                "pH_ideal": [6.8, 8.2],
                "TDS_ideal": [650, 850],
                "volume_ideal": [2, 4],
                "tolerancias": {"temp": [25, 32], "pH": [6.5, 8.5]}
            }
        ])
        print("Estágios básicos inseridos!")

def generate_reading(e, tanque_id, alerta=False):
    params = {
        'temperatura': e['temp_ideal'],
        'pH': e['pH_ideal'],
        'TDS': e['TDS_ideal'],
        'volume': e['volume_ideal']
    }
    
    values = {}
    for param, (low, high) in params.items():
        if not alerta:
            values[param] = np.random.uniform(low, high)
        else:
            tol_low, tol_high = e['tolerancias'].get(param, [low - 2, high + 2])
            if np.random.rand() < 0.5:
                values[param] = np.random.uniform(high, tol_high)
            else:
                values[param] = np.random.uniform(tol_low, low)
    
    return {
        'tanqueId': tanque_id,
        'timestamp': datetime.utcnow() - timedelta(minutes=np.random.randint(0, 1440)),
        **values,
        'faseId': e['_id']
    }

def generate_data(n=35000):
    estagios = list(db.estagios.find())
    tanque_id = "tanque_01"
    
    readings = []
    for _ in range(n):
        for e in estagios:
            alerta = np.random.rand() < 0.2
            readings.append(generate_reading(e, tanque_id, alerta))
    
    db.leituras.insert_many(readings)
    print(f"Inseridas {len(readings)} leituras")

if __name__ == "__main__":
    populate_estagios()
    time.sleep(5)  # Espera MongoDB inicializar
    generate_data(500)
    print("Dados gerados com sucesso!")
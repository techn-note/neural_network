import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import argparse

# Conexão com MongoDB LOCAL
client = MongoClient("mongodb://mongo:27017/")
db = client.piscicultura


ESTAGIOS_DEFAULT = [
    {
        "_id": "alevino",
        "nome": "Alevino",
        "ranges": {
            "temperatura": {"ideal": [28, 30], "tol": [25, 32]},
            "pH":          {"ideal": [7.0, 8.0], "tol": [6.5, 8.5]},
            "TDS":         {"ideal": [600, 800], "tol": [580, 820]},
            "volume":      {"ideal": [1, 2],     "tol": [0.8, 2.2]}
        }
    },
    {
        "_id": "fry",
        "nome": "Fry",
        "ranges": {
            "temperatura": {"ideal": [27, 29], "tol": [25, 32]},
            "pH":          {"ideal": [6.8, 8.2], "tol": [6.5, 8.5]},
            "TDS":         {"ideal": [650, 850], "tol": [630, 870]},
            "volume":      {"ideal": [2, 4],     "tol": [1.5, 4.5]}
        }
    },
    {
        "_id": "finger",
        "nome": "Fingerling",
        "ranges": {
            "temperatura": {"ideal": [26, 28], "tol": [25, 32]},
            "pH":          {"ideal": [6.5, 8.5], "tol": [6.0, 9.0]},
            "TDS":         {"ideal": [700, 900], "tol": [680, 920]},
            "volume":      {"ideal": [4, 6],     "tol": [3.5, 6.5]}
        }
    },
    {
        "_id": "juvenil",
        "nome": "Juvenil",
        "ranges": {
            "temperatura": {"ideal": [25, 28], "tol": [24, 32]},
            "pH":          {"ideal": [6.5, 8.5], "tol": [6.0, 9.0]},
            "TDS":         {"ideal": [800, 1000], "tol": [780, 1020]},
            "volume":      {"ideal": [6, 10],    "tol": [5.0, 12.0]}
        }
    },
    {
        "_id": "engorda",
        "nome": "Engorda",
        "ranges": {
            "temperatura": {"ideal": [25, 27], "tol": [24, 32]},
            "pH":          {"ideal": [6.5, 8.5], "tol": [6.0, 9.0]},
            "TDS":         {"ideal": [0, 2000],  "tol": [0, 2200]},
            "volume":      {"ideal": [50, 200],  "tol": [40, 220]}
        }
    }
]

def populate_estagios():
    est = db.estagios
    if est.estimated_document_count() == 0:
        est.insert_many(ESTAGIOS_DEFAULT)
        print("Estágios básicos inseridos!")

def generate_reading(e, tanque_id, prob_alerta):
    values = {}
    for param, cfg in e['ranges'].items():
        low_i, high_i = cfg['ideal']
        low_t, high_t = cfg['tol']
        if np.random.rand() < prob_alerta:
            # leitura de alerta fora do ideal, mas dentro da tolerância
            if np.random.rand() < 0.5:
                # acima
                values[param] = np.random.uniform(high_i, high_t)
            else:
                # abaixo
                values[param] = np.random.uniform(low_t, low_i)
        else:
            values[param] = np.random.uniform(low_i, high_i)

    return {
        'tanqueId': tanque_id,
        'timestamp': datetime.utcnow() - timedelta(minutes=np.random.randint(0, 1440)),
        **values,
        'faseId': e['_id']
    }


def generate_data(n_por_estagio, prob_alerta, tanque_id="tanque_01"):
    estagios = list(db.estagios.find())
    readings = []
    for e in estagios:
        for _ in range(n_por_estagio):
            readings.append(generate_reading(e, tanque_id, prob_alerta))

    db.leituras.insert_many(readings)
    print(f"Inseridas {len(readings)} leituras ({n_por_estagio} por estágio)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_por_estagio", type=int, default=20000,
                        help="Número de leituras geradas por estágio")
    parser.add_argument("--prob_alerta",  type=float, default=0.2,
                        help="Probabilidade de gerar alerta")
    args = parser.parse_args()

    populate_estagios()
    generate_data(args.n_por_estagio, args.prob_alerta)
    print("Dados gerados com sucesso!")
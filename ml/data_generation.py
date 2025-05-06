import os
import json
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import argparse

# Conexão com MongoDB
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client.piscicultura

# Carrega configuração de estágios a partir de JSON
with open(os.getenv('ESTAGIOS_CONFIG', 'ml/estagios_config.json'), 'r') as f:
    ESTAGIOS = json.load(f)

# Limiares para classificação a partir do score fuzzy
CLASS_THRESHOLDS = {
    'Crise':  (0.0, 0.30),
    'Alerta': (0.31, 0.70),
    'Normal': (0.71, 1.0)
}


def fuzzy_score(x, low_tol, low_ideal, high_ideal, high_tol):
    """
    Calcula um score fuzzy entre 0 e 1 para valor x dado tolerância e ideal.
    """
    if x < low_tol or x > high_tol:
        return 0.0
    if low_ideal <= x <= high_ideal:
        return 1.0
    if x < low_ideal:
        return (x - low_tol) / (low_ideal - low_tol)
    return (high_tol - x) / (high_tol - high_ideal)


def generate_reading(e, tanque_id, prob_alerta):
    """
    Gera uma leitura sintética para um estágio e tanque, com probabilidade de alerta.
    """
    vals = {}
    scores = []

    for param, cfg in e['ranges'].items():
        # Para tds, garantir mínimo >= 1.0 se não especificado
        default_min = 1.0 if param == 'tds_mg_L' else 0.0
        low_t  = cfg.get('min', default_min)
        high_t = cfg.get('max', low_t)
        low_i  = cfg.get('ideal_min', low_t)
        high_i = cfg.get('ideal_max', high_t)

        # Validação de consistência dos limites
        if not (low_t <= low_i <= high_i <= high_t):
            raise ValueError(f"Limites inconsistentes em {param}: {cfg}")

        # Ajuste na geração de valores para garantir uma distribuição mais equilibrada entre as classes
        # Adicionando um controle para ajustar a probabilidade de valores ideais e de alerta
        if np.random.rand() < prob_alerta:
            if np.random.rand() < 0.5:
                v = np.random.uniform(high_i, high_t)
            else:
                v = np.random.uniform(low_t, low_i)
        else:
            v = np.random.uniform(low_i, high_i)

        # Garantir que os valores gerados sejam consistentes com os limiares definidos
        v = max(min(v, high_t), low_t)
        vals[param] = v

        # Computa score fuzzy
        scores.append(fuzzy_score(v, low_t, low_i, high_i, high_t))

    # Ajuste na lógica de classificação para garantir que os rótulos sejam atribuídos corretamente
    score = float(np.mean(scores))
    label = None
    for c, (lo, hi) in CLASS_THRESHOLDS.items():
        if lo <= score <= hi:
            label = c
            break
    if label is None:
        raise ValueError(f"Score {score} fora dos limites definidos em CLASS_THRESHOLDS")

    # Mapeia chaves para inserção consistente
    return {
        'tanqueId': tanque_id,
        'timestamp': datetime.utcnow() - timedelta(minutes=np.random.randint(0, 1440)),
        'temperatura': vals['temperatura'],
        'ph':          vals['ph'],
        'tds':         vals['tds'],
        'faseId':      e['estagio'],
        'class':       label
    }


def generate_data(n_por_estagio, prob_alerta, tanque_id="tanque_01"):
    """
    Gera n_por_estagio leituras por estágio e insere no MongoDB.
    """
    readings = []
    for e in ESTAGIOS:
        for _ in range(n_por_estagio):
            readings.append(generate_reading(e, tanque_id, prob_alerta))
    db.leituras.insert_many(readings)
    print(f"Inseridas {len(readings)} leituras ({n_por_estagio} por estágio)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_por_estagio", type=int, default=100)
    parser.add_argument("--prob_alerta", type=float, default=0.3)
    args = parser.parse_args()

    generate_data(args.n_por_estagio, args.prob_alerta)
    print("Geração de dados concluída.")
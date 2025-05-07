import os
import json
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import argparse
from collections import defaultdict
from typing import Tuple, Dict

# Configuração do MongoDB
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client.piscicultura

# Carregar configuração de estágios
with open(os.getenv('ESTAGIOS_CONFIG', 'estagios_config.json'), 'r') as f:
    ESTAGIOS = json.load(f)

CLASS_THRESHOLDS = {
    'Crise': (0.0, 0.3),
    'Anormal': (0.3, 0.7),
    'Normal': (0.7, 1.0)
}

def ajustar_faixas(cfg):
    """Garante uma variação mínima para cálculo do score"""
    for param in cfg.values():
        if param['ideal_min'] == param['min']:
            param['min'] = param['ideal_min'] - 0.1
        if param['ideal_max'] == param['max']:
            param['max'] = param['ideal_max'] + 0.1
    return cfg

def calcular_score(vals, cfg):
    """Calcula o score total garantindo os limiares especificados"""
    scores = []
    for param in ['temperatura', 'ph', 'tds']:
        v = vals[param]
        p = cfg[param]
        
        if v < p['min'] or v > p['max']:
            score = 0.0
        elif p['ideal_min'] <= v <= p['ideal_max']:
            score = 1.0
        else:
            if v < p['ideal_min']:
                score = (v - p['min']) / (p['ideal_min'] - p['min'])
            else:
                score = (p['max'] - v) / (p['max'] - p['ideal_max'])
        
        scores.append(score)
    
    return round(np.mean(scores), 2)

def gerar_dados_por_classe(
    classe_alvo: str,
    cfg: dict,
    thresholds: dict,
    max_tentativas: int = 50,
    random_state: int = None
) -> Tuple[dict, float]:
    if random_state is not None:
        np.random.seed(random_state)

    min_th, max_th = thresholds[classe_alvo]

    for _ in range(max_tentativas):
        vals = {}
        for param, p in cfg.items():
            u = np.random.rand()
            if classe_alvo == 'Crise':
                low, high = (p['min'], p['ideal_min']) if u < 0.5 else (p['ideal_max'], p['max'])
            elif classe_alvo == 'Anormal':
                low, high = (p['min'], p['ideal_min']) if u < 0.5 else (p['ideal_max'], p['max'])
            else:  # Normal
                low, high = p['ideal_min'], p['ideal_max']

            vals[param] = np.random.uniform(low, high)

        score = calcular_score(vals, cfg)
        if min_th <= score <= max_th:
            return vals, score

    # Se não conseguiu em X tentativas, retorna o melhor encontrado:
    return vals, score


def generate_dataset(total_amostras, distribuicao):
    """Gera dataset com distribuição controlada"""
    leituras = []
    contador = defaultdict(int)
    
    for classe, porcentagem in distribuicao.items():
        if porcentagem <= 0:
            continue
        
        n = int(total_amostras * porcentagem)
        for _ in range(n):
            estagio = np.random.choice(ESTAGIOS)
            cfg = ajustar_faixas(estagio['ranges'].copy())
            
            vals, score = gerar_dados_por_classe(classe, cfg, CLASS_THRESHOLDS, random_state=None)
            
            leituras.append({
                'temperatura': round(vals['temperatura'], 2),
                'ph': round(vals['ph'], 2),
                'tds': round(vals['tds'], 2),
                'faseId': estagio['estagio'],
                'class': classe,
            })
            contador[classe] += 1
    
    # Inserir no MongoDB
    if leituras:
        db.leituras.insert_many(leituras)
    
    return contador

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gerador de Dataset para Monitoramento Aquícola')
    parser.add_argument('--total', type=int, default=2000000, help='Total de amostras')
    parser.add_argument('--normal', type=float, default=0.65, help='Porcentagem de casos Normais')
    parser.add_argument('--anormal', type=float, default=0.20, help='Porcentagem de casos Anormais')
    parser.add_argument('--crise', type=float, default=0.15, help='Porcentagem de casos de Crise')
    
    args = parser.parse_args()
    
    total = args.normal + args.anormal + args.crise
    if not np.isclose(total, 1.0, atol=0.01):
        raise ValueError("A soma das porcentagens deve ser 100% (1.0 no total)")
    
    distribuicao = {
        'Normal': args.normal,
        'Anormal': args.anormal,
        'Crise': args.crise
    }
    
    print(f"\n🚀 Iniciando geração de {args.total} amostras:")
    print(f"| {'Classe':<10} | {'Amostras':<8} |")
    print("|------------|----------|")
    for classe, pct in distribuicao.items():
        print(f"| {classe:<10} | {int(args.total*pct):<8} |")
    
    contagem = generate_dataset(args.total, distribuicao)
    
    print("\n✅ Distribuição final:")
    print(f"| {'Classe':<10} | {'Count':<6} | {'%':<6} |")
    print("|------------|--------|-------|")
    total_gerado = sum(contagem.values())
    for classe, count in contagem.items():
        print(f"| {classe:<10} | {count:<6} | {count/total_gerado:.1%} |")
    
    print(f"\n💾 Dados salvos no MongoDB")
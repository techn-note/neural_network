# analysis.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Configuração de visualização
plt.style.use('ggplot')
sns.set_palette("husl")
os.makedirs('plots', exist_ok=True)

# 1. Carregar artefatos
print("Carregando artefatos...")
scaler = joblib.load('/app/artifacts/scaler.pkl')
ohe = joblib.load('/app/artifacts/ohe.pkl')
X = np.load('/app/artifacts/X.npy')
y = np.load('/app/artifacts/y.npy')
model = load_model('/app/artifacts/modelo_tilapia.h5')

# 2. Gerar previsões
print("Gerando previsões...")
y_pred = model.predict(X)

# 3. Métricas por parâmetro
parametros = ['Temperatura', 'pH', 'TDS', 'Volume']
print("\nMétricas por parâmetro:")
for i in range(4):
    mse = mean_squared_error(y[:, i], y_pred[:, i])
    r2 = r2_score(y[:, i], y_pred[:, i])
    print(f"{parametros[i]}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("─" * 50)

# 4. Gráficos de dispersão
plt.figure(figsize=(15, 10))
for i, param in enumerate(parametros):
    plt.subplot(2, 2, i+1)
    sns.regplot(x=y[:, i], y=y_pred[:, i], line_kws={'color': 'red'})
    plt.title(f'True vs Predito - {param}')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/dispersao.png')
plt.close()

# 5. Distribuição de resíduos
plt.figure(figsize=(15, 10))
for i, param in enumerate(parametros):
    plt.subplot(2, 2, i+1)
    residuals = y[:, i] - y_pred[:, i]
    sns.histplot(residuals, kde=True)
    plt.title(f'Distribuição de Resíduos - {param}')
    plt.xlabel('Erro')
plt.tight_layout()
plt.savefig('plots/residuos.png')
plt.close()

# 6. Importância das features (via permutação)
print("\nCalculando importância das features...")
feature_names = [
    'Temperatura', 'pH', 'TDS', 'Volume',
    *ohe.get_feature_names_out(['faseId']),
    'sin_hora', 'cos_hora'
]

result = permutation_importance(
    model, X, y,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

# Plot
plt.figure(figsize=(12, 8))
sorted_idx = result.importances_mean.argsort()
plt.barh(
    np.array(feature_names)[sorted_idx],
    result.importances_mean[sorted_idx],
    xerr=result.importances_std[sorted_idx]
)
plt.title('Importância das Features por Permutação')
plt.tight_layout()
plt.savefig('plots/importancia_features.png')
plt.close()

# 7. Evolução do treino (se history.npy existir)
try:
    history = np.load('/app/artifacts/history.npy', allow_pickle=True).item()
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Treino')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Evolução da Loss durante o Treino')
    plt.ylabel('MSE')
    plt.xlabel('Época')
    plt.legend()
    plt.savefig('plots/evolucao_treino.png')
    plt.close()
except FileNotFoundError:
    print("Arquivo history.npy não encontrado - pulando plot de evolução")

print("\nAnálise concluída! Gráficos salvos na pasta /plots")
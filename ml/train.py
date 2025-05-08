import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from model import build_model
from collections import Counter

# Configuração de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Carrega dados
X = np.load('artifacts/X.npy')
y = np.load('artifacts/y.npy')

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
)

# Verifica a distribuição das classes após o split
def print_class_distribution(y, dataset_name):
    class_counts = Counter(np.argmax(y, axis=1))
    print(f"Distribuição de classes em {dataset_name}:")
    for class_label, count in class_counts.items():
        print(f"Classe {class_label}: {count} amostras")

print_class_distribution(y_train, "Treinamento")
print_class_distribution(y_val, "Validação")

# Cria modelo
model = build_model(input_dim=X_train.shape[1], num_classes=y.shape[1])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('artifacts/best_model.h5', save_best_only=True)
]

# Treina
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks
)

# Avaliação final
val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

# Classification report e F1
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_val, axis=1)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Crise","Anormal","Normal"]))
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1-score: {f1:.4f}")

# Salva artefatos
model.save('artifacts/safira.h5')
np.save('artifacts/history.npy', history.history)
print("Treino concluído e artefatos salvos em artifacts/")
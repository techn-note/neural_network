import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model
import os

# GPU config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# carrega dados pré-processados
X = np.load('/app/artifacts/X.npy')
y = np.load('/app/artifacts/y.npy')  # shape (n_amostras, 4)

# split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# cria modelo com a nova arquitetura
model = build_model(X_train.shape[1])

# callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('/app/artifacts/best_model.h5', save_best_only=True)
]

# treina
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks
)

# salva artefatos
model.save('/app/artifacts/modelo_tilapia.h5')
np.save('/app/artifacts/history.npy', history.history)
print("Treino concluído, artefatos salvos.")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model
import joblib
import os

# Carrega dados
X = np.load('X.npy')
y = np.load('y.npy')

# Divisão treino/teste
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y, 
    random_state=42
)

# Construção do modelo
model = build_model(X_train.shape[1])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Treinamento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Salva modelo
model.save('modelo_tilapia.h5')
print("Modelo treinado e salvo!")
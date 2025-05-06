import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib, tensorflow as tf
sns.set()

# Load artifacts and history
X=np.load('artifacts/X.npy'); y=np.load('artifacts/y.npy'); hist=np.load('artifacts/history.npy',allow_pickle=True).item()
model=tf.keras.models.load_model('artifacts/fish_quality_classifier.h5')

# Split train/test inside test script
from sklearn.model_selection import train_test_split
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# 1. Loss & Accuracy Curves
plt.figure(); plt.plot(hist['loss'],label='train_loss'); plt.plot(hist['val_loss'],label='val_loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.savefig('artifacts/analysis/loss_curve.png'); plt.close()
plt.figure(); plt.plot(hist['accuracy'],label='train_acc'); plt.plot(hist['val_accuracy'],label='val_acc'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.savefig('artifacts/analysis/acc_curve.png'); plt.close()

# 2. Confusion Matrix
y_pred=np.argmax(model.predict(Xte),axis=1); y_true=np.argmax(yte,axis=1)
cm=confusion_matrix(y_true,y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Crise','Alerta','Normal'],yticklabels=['Crise','Alerta','Normal']); plt.ylabel('True'); plt.xlabel('Pred'); plt.savefig('artifacts/analysis/confusion_matrix.png'); plt.close()

# 3. Classification Report
print('Classification Report:\n',classification_report(y_true,y_pred,target_names=['Crise','Alerta','Normal']))

# 4. Parameter Score Distribution
# if computed in artifacts, else skip

print('Analysis plots saved in artifacts/analysis/')
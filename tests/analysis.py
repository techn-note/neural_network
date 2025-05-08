import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import joblib, tensorflow as tf
sns.set()

# Load artifacts and history
X=np.load('artifacts/X.npy'); y=np.load('artifacts/y.npy'); hist=np.load('artifacts/history.npy',allow_pickle=True).item()
model=tf.keras.models.load_model('artifacts/safira.h5')

# Split train/test inside test script
from sklearn.model_selection import train_test_split
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Garantir que o diretório artifacts/analysis/ exista
output_dir = 'artifacts/analysis/'
os.makedirs(output_dir, exist_ok=True)

# 1. Loss & Accuracy Curves
plt.figure(); plt.plot(hist['loss'],label='train_loss'); plt.plot(hist['val_loss'],label='val_loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve'); plt.savefig(os.path.join(output_dir, 'loss_curve.png')); plt.close()
plt.figure(); plt.plot(hist['accuracy'],label='train_acc'); plt.plot(hist['val_accuracy'],label='val_acc'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve'); plt.savefig(os.path.join(output_dir, 'acc_curve.png')); plt.close()

# 2. Confusion Matrix
y_pred=np.argmax(model.predict(Xte),axis=1); y_true=np.argmax(yte,axis=1)
cm=confusion_matrix(y_true,y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Crise','Anormal','Normal'],yticklabels=['Crise','Anormal','Normal']); plt.ylabel('True'); plt.xlabel('Pred'); plt.savefig(os.path.join(output_dir, 'confusion_matrix.png')); plt.close()

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6,5)); sns.heatmap(cm_normalized,annot=True,fmt='.2f',cmap='Blues',xticklabels=['Crise','Anormal','Normal'],yticklabels=['Crise','Anormal','Normal']); plt.ylabel('True'); plt.xlabel('Pred'); plt.title('Normalized Confusion Matrix'); plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png')); plt.close()

# 3. Classification Report
print('Classification Report:\n',classification_report(y_true,y_pred,target_names=['Crise','Anormal','Normal']))

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f'Balanced Accuracy: {balanced_acc:.2f}')

# 4. ROC and AUC Curves
classes = ['Crise', 'Anormal', 'Normal']
yte_binarized = label_binarize(yte, classes=[0, 1, 2])
n_classes = yte_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(yte_binarized[:, i], model.predict(Xte)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
for i, class_name in enumerate(classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {class_name}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Precision-Recall Curve
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(yte_binarized[:, i], model.predict(Xte)[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curves
plt.figure()
for i, class_name in enumerate(classes):
    plt.plot(recall[i], precision[i], label=f'PR curve (area = {pr_auc[i]:.2f}) for class {class_name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
plt.close()

print('Analysis plots saved in artifacts/analysis/')
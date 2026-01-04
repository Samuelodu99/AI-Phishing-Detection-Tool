import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
# Example: y_true = true labels, y_score = model probabilities
y_true = np.array([1]*50 + [0]*50)  # 50 phishing, 50 legitimate
y_score = np.random.rand(100)  # Replace with actual XGBoost probabilities
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Classifier')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()
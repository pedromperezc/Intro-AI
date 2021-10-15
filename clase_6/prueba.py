import numpy as np
from models import LogisticRegressionNumpy
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier



with open('X_train.npy', 'rb') as f:
    X_train = np.load(f)

with open('y_train.npy', 'rb') as f:
    y_train = np.load(f)

with open('X_test.npy', 'rb') as f:
    X_test = np.load(f)

with open('y_test.npy', 'rb') as f:
    y_test = np.load(f)


# dataset = Data('clase_6_dataset.txt')
# X_train, X_test, y_train, y_test = dataset.split(0.8)


model = BalancedRandomForestClassifier()

model.fit(X_train, y_train)


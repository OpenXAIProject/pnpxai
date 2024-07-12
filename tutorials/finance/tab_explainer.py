import pickle

import torch
import numpy as np
import pandas as pd
import quantus as qt
import qt_wrapper as qtw
import seaborn as sns
import matplotlib.pyplot as plt

from pnpxai.explainers import TabKernelShap, TabLime
from shap.utils._legacy import kmeans
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import sklearn

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
# Load train and test data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.80)

clf_model = XGBClassifier()
clf_model.fit(X_train, y_train)

categorical_features = np.argwhere(np.array([len(set(X_train[:,x])) for x in range(X_train.shape[1])]) <= 10).flatten()
bg_data = X_train
explainer = TabLime(
    clf_model, bg_data, categorical_features=categorical_features, mode='classification'
)
attribution = explainer.attribute(
    X_test[:10],
    targets=None,
    n_samples=1000)

print(attribution)

bg_data = TabKernelShap.kmeans(X_train, 100)
explainer = TabKernelShap(clf_model, bg_data, mode="classification")
targets = clf_model.predict(X_test[:10])
attribution = explainer.attribute(
    X_test[:10],
    targets=targets,
    n_samples=1000)

print(attribution)


reg_model = XGBRegressor()
housing = fetch_california_housing()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(housing.data, housing.target, train_size=0.80)
reg_model.fit(train, labels_train)


categorical_features = np.argwhere(np.array([len(set(train[:,x])) for x in range(train.shape[1])]) <= 10).flatten()
bg_data = train
explainer = TabLime(
    reg_model, bg_data, categorical_features=categorical_features, mode='regression'
)
attribution = explainer.attribute(
    test[:10],
    targets=None,
    n_samples=1000)

print(attribution)

bg_data = TabKernelShap.kmeans(train, 100)
explainer = TabKernelShap(reg_model, bg_data, mode="regression")
attribution = explainer.attribute(
    test[:10],
    targets=None,
    n_samples=1000)

print(attribution)



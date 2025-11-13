import os
import pickle
import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from models.tab_resnet import TabResNet, LogisticRegression

def invert_input_array(input_array, feature_metadata):
    inverted_data = {}
    
    for col, meta in feature_metadata.items():
        if meta['type'] == 'categorical':
            # One-hot encoded 된 부분 추출
            start_idx, end_idx = meta['index'][0], meta['index'][-1] + 1
            cat_data = input_array[:, start_idx:end_idx]
            # OneHotEncoder로 복원
            inverted_col = meta['encoder'].inverse_transform(cat_data)
            inverted_data[col] = inverted_col.flatten()
        else:
            # 수치형 데이터 복원
            idx = meta['index']
            num_data = input_array[:, idx].reshape(-1, 1)
            inverted_col = meta['encoder'].inverse_transform(num_data)
            inverted_data[col] = inverted_col.flatten()
    
    # 복원된 데이터를 DataFrame으로 변환
    inverted_df = pd.DataFrame(inverted_data)
    
    return inverted_df



dataset_id = 186
dataset = fetch_ucirepo(id=dataset_id)

# feature_metadata = {}
# for col in dataset.data.features.columns:
#     feature_metadata[col] = {}
#     if dataset.data.features[col].dtype == "object":
#         feature_metadata[col]['type'] = "categorical"
#     else:
#         feature_metadata[col]['type'] = "numerical"
        
# feature_metadata

feature_metadata = {}
input_data = []
start_idx = 0
for col in dataset.data.features.columns:
    feature_metadata[col] = {}
    if dataset.data.features[col].dtype == "object":
        feature_metadata[col]['type'] = "categorical"
        onehot = OneHotEncoder(handle_unknown='ignore')
        feature_val = dataset.data.features[col].fillna("missing")
        preprocessed = onehot.fit_transform(feature_val.values.reshape(-1, 1)).toarray()
        cat_dist = feature_val.value_counts(dropna=False) / len(dataset.data.features)
        cat_dist = cat_dist.loc[onehot.categories_[0]].values
        feature_metadata[col]['encoder'] = onehot
        feature_metadata[col]['cat_dist'] = cat_dist
        feature_metadata[col]['index'] = np.arange(start_idx, start_idx + preprocessed.shape[1])
        start_idx += preprocessed.shape[1]
    else:
        feature_metadata[col]['type'] = "numerical"
        scaler = StandardScaler()
        preprocessed = scaler.fit_transform(dataset.data.features[col].values.reshape(-1, 1))
        feature_metadata[col]['encoder'] = scaler
        feature_metadata[col]['index'] = start_idx
        start_idx += 1

    input_data.append(preprocessed)

print(feature_metadata)

input_array = np.concatenate(input_data, axis=1)

print(dataset.data.targets.isin([7,8,9]).value_counts() / len(dataset.data.targets))
y = dataset.data.targets.isin([7,8,9]).values.astype(int)[:,0]

X_train, X_test, y_train, y_test = train_test_split(input_array, y, test_size=0.2, random_state=42, stratify=y)

path = f"data/{dataset.metadata['name']}"
if not os.path.exists(path):
    os.makedirs(path)

np.save(f"{path}/X_train.npy", X_train)
np.save(f"{path}/X_test.npy", X_test)
np.save(f"{path}/y_train.npy", y_train)
np.save(f"{path}/y_test.npy", y_test)

with open(f"{path}/feature_metadata.pkl", "wb") as f:
    pickle.dump(feature_metadata, f)


# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

dataset = "Wine Quality"
path = f"data/{dataset}"
X_train = np.load(f"{path}/X_train.npy")
y_train = np.load(f"{path}/y_train.npy")
X_test = np.load(f"{path}/X_test.npy")
y_test = np.load(f"{path}/y_test.npy")

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

xgb_y_pred = xgb_clf.predict(X_test)
xgb_accuracy = np.mean(xgb_y_pred == y_test)
print(f"XGBoost Accuracy: {xgb_accuracy}")

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()

feature_metadata = pickle.load(open(f"{path}/feature_metadata.pkl", "rb"))
xgb_clf.save_model(f"{path}/xgb_model.json")

input_dim = X_train.shape[1]
output_dim = 2

resnet_model = TabResNet(input_dim, output_dim, num_blocks=1)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.01, weight_decay=0.01)

train_model(X_train, y_train, resnet_model, loss_fn, optimizer, 1000)

resnet_y_pred = resnet_model(X_test).detach().argmax(dim=1).numpy()
resnet_accuracy = np.mean(resnet_y_pred == y_test)
print(f"ResNet Accuracy: {resnet_accuracy}")

torch.save(lr_model.state_dict(), f"{path}/lr_model.pth")
torch.save(resnet_model.state_dict(), f"{path}/resnet_model.pth")
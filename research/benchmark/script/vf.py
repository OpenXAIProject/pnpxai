import os
import random
import functools
import argparse
import logging
import pickle
from math import comb

import torch
import pandas as pd
import numpy as np
import xgboost as xgb

from models.tab_resnet import TabResNet

from sklearn.cluster import KMeans


def _transform(X, feature_metadata):
    input_data = []
    for k, v in feature_metadata.items():
        preprocessed = v['encoder'].transform(X[[k]].values)
        if v['type'] == 'categorical':
            preprocessed = preprocessed.toarray()
        input_data.append(preprocessed)
    
    input_array = np.concatenate(input_data, axis=1)
    return input_array

def _invert_input_array(input_array, feature_metadata):
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

def find_closest_data_with_center(X_train, cluster_centers):
    closest_data = []
    
    for center in cluster_centers:
        # 각 중심에 대해 유클리드 거리 계산
        distances = np.linalg.norm(X_train - center, axis=1)
        # 가장 가까운 데이터의 인덱스 찾기
        closest_index = np.argmin(distances)
        # 가장 가까운 데이터 추가
        closest_data.append(X_train[closest_index])
    
    return np.array(closest_data)

def vf_converter(X_test, bg_data, model, transform, weight):
    orig_feature = bg_data.columns.copy()
    target = invert_input_array(X_test)
    proba = model.predict_proba(transform(target))
    pred_label = proba.argmax()

    def value_function(bin_coal):
        coalition = np.where(bin_coal == 1)[0]
        coal_feature = orig_feature[coalition]
        non_coal_feature = [col for col in orig_feature if col not in coal_feature]

        coal_data = target.loc[target.index.repeat(len(bg_data)), coal_feature].reset_index(drop=True)
        non_coal_data = bg_data.loc[np.tile(bg_data.index, len(target))].reset_index(drop=True)[non_coal_feature]
        new_data = pd.concat([coal_data, non_coal_data], axis=1)[orig_feature]

        input_data = transform(new_data)
        pred = model.predict_proba(input_data)[:, pred_label]

        return (pred * weight).sum()
    return value_function


def validate_args(args):
    if args.dataset not in ["Adult", "Bank Marketing", "Statlog (German Credit Data)", "Wine Quality"]:
        raise ValueError("Invalid dataset")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Adult")
    parser.add_argument("--model", type=str, default="xgb")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    args = parser.parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    dataset_nm = args.dataset
    X_train = np.load(f"data/{dataset_nm}/X_train.npy")
    y_train = np.load(f"data/{dataset_nm}/y_train.npy")

    X_test = np.load(f"data/{dataset_nm}/X_test.npy")
    y_test = np.load(f"data/{dataset_nm}/y_test.npy")

    feature_metadata = pickle.load(open(f"data/{dataset_nm}/feature_metadata.pkl", "rb"))
    raw_data = pd.read_csv(f"data/{dataset_nm}/raw_data.csv")
    invert_input_array = functools.partial(_invert_input_array, feature_metadata=feature_metadata)
    transform = functools.partial(_transform, feature_metadata=feature_metadata)

    kmeans = KMeans(n_clusters=50, random_state=42)
    kmeans.fit(X_train)

    res = kmeans.predict(X_train)
    _, counts = np.unique(res, return_counts=True)
    weight = counts / counts.sum()

    bg_data = find_closest_data_with_center(X_train, kmeans.cluster_centers_)
    bg_data = invert_input_array(bg_data)
    train_data = invert_input_array(X_train)

    if args.model == "xgb":
        model = xgb.XGBClassifier()
        model.load_model(f"data/{dataset_nm}/xgb_model.json")

    elif args.model == "tab_resnet":
        model = TabResNet(X_train.shape[1], 2, num_blocks=1)
        model.load_state_dict(torch.load(f"data/{dataset_nm}/resnet_model.pth"))
        model.eval()

    path = f"results/{dataset_nm}/{args.model}/{args.framework}/{args.explainer}"
    explanation = np.load(f"{path}/explanations.npy")

    n = len(feature_metadata)
    input_vector = np.array([[int(x) for x in format(i, f'0{n}b')] for i in range(2**n)])

    data_size = 10
    vf_values = np.zeros((data_size, 2**n))
    for i in tqdm(range(data_size)):
        vf = vf_converter(X_test[[i]], bg_data, model, transform, weight)

        vf_values[i] = np.array([vf(x) for x in input_vector])
    
    np.save(f"{path}/vf_value.npy", vf_values)
    logging.info(f"Saved vf values to {path}/vf_value.npy")

# python script/evaluate.py --dataset "Wine Quality" --model tab_resnet --framework openxai --explainer lime --metric abpc 
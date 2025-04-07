import argparse
import os
import random
import functools
import pickle
from collections import defaultdict, deque

import torch
import pandas as pd
import numpy as np
import xgboost as xgb

def find_idx(a, b):
    """
    두 리스트 a, b가 주어졌을 때,
    어떤 순열 idx로 a[idx] = b를 만족시키는 idx를 구한다.
    만약 불가능하면 None (또는 예외) 반환.
    """
    # 1) a, b가 같은 multiset인지 확인
    if sorted(a) != sorted(b):
        return None  # 혹은 raise ValueError("a와 b가 같은 요소를 갖고 있지 않습니다.")

    # 2) a에 대해 '값 -> 해당 값의 인덱스 리스트'를 만든다.
    #    중복 값을 처리하기 위해 deque로 관리
    pos_map = defaultdict(deque)
    for i, val in enumerate(a):
        pos_map[val].append(i)

    # 3) b를 순회하며, 각 원소에 매핑되는 a의 인덱스를 하나씩 꺼내 idx를 구성
    idx = []
    for val in b:
        idx.append(pos_map[val].popleft())

    return idx

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Adult")
    parser.add_argument("--model", type=str, default="xgb")
    parser.add_argument("--framework", type=str, default="omnixai")
    parser.add_argument("--explainer", type=str, default="lime")
    args = parser.parse_args()

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

    if args.model == "xgb":
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f"data/{dataset_nm}/xgb_model.json")

    elif args.model == "ann":
        # nn_model = torch.nn.Sequential(
        #     torch.nn.Linear(X_train.shape[1], 3),
        #     torch.nn.Softmax(dim=1)
        # )

        # model = torch.load(f"data/{dataset_nm}/ann_model.pth")

        raise NotImplementedError

    if args.framework == "omnixai":
        from omnixai.data.tabular import Tabular
        from omnixai.explainers.tabular import TabularExplainer

        NAME_MAP = {
            "lime" : "LimeTabular",
            "shap" : "ShapTabular"
        }

        explainer_nm = NAME_MAP[args.explainer]

        train_data = Tabular(raw_data, categorical_columns=[c for c in raw_data.columns if feature_metadata[c]["type"] == "categorical"])
        transform = functools.partial(_transform, feature_metadata=feature_metadata)
        invert_input_array = functools.partial(_invert_input_array, feature_metadata=feature_metadata)

        if args.model == "xgb":
            target_function = xgb_model.predict_proba


        explainer = TabularExplainer(
        explainers=[explainer_nm],
        mode="classification",                             # The task type
        data=train_data,
        model=target_function,
        preprocess=lambda z: transform(z.data),
        )

        test_instances = invert_input_array(X_test[:3])
        # test_instances = invert_input_array(X_test)

        params = {}
        if explainer_nm == "LimeTabular":
            params = {
                "LimeTabular" : {"num_features": raw_data.shape[1]}
            }
        exp_obj = explainer.explain(test_instances, params=params)
        scores = []
        for i in range(test_instances.shape[0]):
            exp = exp_obj[explainer_nm].get_explanations(i)
            sorted_idx = find_idx(exp['features'], exp['instance'].columns.tolist())
            scores.append([exp['scores'][i] for i in sorted_idx])

        explanations = np.array(scores)

# save explanation
path = f"results/{dataset_nm}/{args.model}/{args.framework}/{args.explainer}"
if not os.path.exists(path):
    os.makedirs(path)

np.save(f"{path}/explanations.npy", explanations)


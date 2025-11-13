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
from sklearn.cluster import KMeans
import torch.nn as nn
from torch.utils.data import DataLoader

from pnpxai import Experiment, AutoExplanation
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers import Lime, KernelShap
from pnpxai.evaluator.metrics import AbPC, Metric, Complexity

from models.tab_resnet import TabResNet

class TorchModelForXGBoost(nn.Module):
    def __init__(self, xgb_model):
        super().__init__()
        self.xgb_model = xgb_model
        self._dummy_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor):
        if x.ndim >= 3:
            x = x.squeeze(0)
        out = self.xgb_model.predict_proba(x.cpu().numpy())
        return torch.from_numpy(out)

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

def shapley_kernel(N):
    kernel = np.zeros(N)
    for s in range(1, N):  # subset 크기 s는 1부터 N-1까지
        kernel[s] = (N - 1) / (comb(N, s) * s * (N - s))
    return kernel / kernel.sum()


def create_masked_vector(explanation, d):
    """
    Create a masked binary vector for MORF or LERF calculation.
    """
    sorted_idx = np.argsort(-explanation)  # 내림차순 정렬
    bin_vec = np.ones((d + 1, d), dtype=int)
    row_indices = np.broadcast_to(np.arange(d + 1)[:, None], (d + 1, d))
    sorted_indices = np.broadcast_to(sorted_idx, (d + 1, d))
    mask = (np.arange(d)[None, :] < np.arange(d + 1)[:, None])
    bin_vec[row_indices[mask], sorted_indices[mask]] = 0
    return bin_vec

def compute_curve(bin_vec, target, bg_data, model, transform, pred_label, weight):
    """
    Compute MORF or LERF curve based on the masked vector.
    """
    orig_feature = bg_data.columns.copy()
    curve = np.zeros(bin_vec.shape[0])
    
    for i, bin_coal in enumerate(bin_vec):
        coalition = np.where(bin_coal == 1)[0]
        coal_feature = orig_feature[coalition]
        non_coal_feature = [col for col in orig_feature if col not in coal_feature]

        coal_data = target.loc[target.index.repeat(len(bg_data)), coal_feature].reset_index(drop=True)
        non_coal_data = bg_data.loc[np.tile(bg_data.index, len(target))].reset_index(drop=True)[non_coal_feature]
        new_data = pd.concat([coal_data, non_coal_data], axis=1)[orig_feature]

        input_data = transform(new_data)
        pred = model.predict_proba(input_data)[:, pred_label]

        # 곡선 업데이트
        curve[i] = pred.reshape(-1, bg_data.shape[0]) @ weight

    return curve

def compute_abpc(X_test, bg_data, model, transform, explanation, weight):
    """
    Compute ABPC based on MORF and LERF curves.
    """
    target = invert_input_array(X_test)
    proba = model.predict_proba(transform(target))
    pred_label = proba.argmax()

    # MORF 곡선 생성
    morf_bin_vec = create_masked_vector(explanation, len(bg_data.columns))
    morf_curve = compute_curve(morf_bin_vec, target, bg_data, model, transform, pred_label, weight)

    # LERF 곡선 생성
    lerf_bin_vec = create_masked_vector(-explanation, len(bg_data.columns))
    lerf_curve = compute_curve(lerf_bin_vec, target, bg_data, model, transform, pred_label, weight)

    if pred_label == 0:
        morf_curve, lerf_curve = lerf_curve, morf_curve

    # ABPC 계산
    abpc = (lerf_curve - morf_curve).mean()
    return abpc, morf_curve, lerf_curve


def validate_args(args):
    args.dataset = args.dataset.replace('\\','')
    if args.dataset not in ["Adult", "Bank Marketing", "Statlog (German Credit Data)", "Wine Quality"]:
        raise ValueError("Invalid dataset")
    
    if args.framework not in ["omnixai", "openxai", "pnpxai", "autoxai", "captum"]:
        raise ValueError("Invalid framework")
    
    if args.framework != "pnpxai" and args.explainer not in ["lime", "shap", "ig", "grad", "sg", "itg", "lrp", "vg"]:
        raise ValueError("Invalid explainer")
    
    if args.framework == "omnixai" and args.model != "xgb":
        raise ValueError("omnixai framework only supports xgb model")
    
    if args.framework == "openxai" and args.model == "xgb":
        raise ValueError("openxai framework does not support xgb model")    

    if args.framework == "omnixai" and args.explainer not in ["lime", "shap"]:
        raise ValueError("omnixai framework only supports lime and shap explainer")

    if args.framework == "openxai" and args.explainer in ["lrp", "vg"]:
        raise ValueError("openxai framework does not supports lrp and vg explainer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Adult")
    parser.add_argument("--model", type=str, default="xgb")
    parser.add_argument("--framework", type=str, default="omnixai")
    parser.add_argument("--explainer", type=str, default="lime")
    parser.add_argument("--metric", type=str, default="abpc")

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

    if args.model == "xgb":
        model = xgb.XGBClassifier()
        model.load_model(f"data/{dataset_nm}/xgb_model.json")

    elif args.model == "tab_resnet":
        model = TabResNet(X_train.shape[1], 2, num_blocks=1)
        model.load_state_dict(torch.load(f"data/{dataset_nm}/resnet_model.pth"))
        model.eval()

    test_input = torch.tensor(X_test, dtype=torch.float32)
    test_loader = DataLoader(
        test_input,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
    )

    if args.model == "xgb":
        model = TorchModelForXGBoost(xgb_model=model)

    sample_batch = next(iter(test_loader))
    modality = Modality(
        dtype=sample_batch[0].dtype,
        ndims=2,
        pooling_dim=1
    )

    class CompoundMetric(Metric):
        def __init__(
            self,
            model,
            cmpd_metrics,
            weights, 
            explainer=None,
            target_input_keys=None,
            additional_input_keys=None,
            output_modifier=None,
        ):
            super().__init__(
                model, explainer, target_input_keys,
                additional_input_keys, output_modifier,
            )
            assert len(cmpd_metrics) == len(weights)
            self.cmpd_metrics = cmpd_metrics
            self.weights = weights

        def evaluate(self, inputs, targets, attrs):
            values = torch.zeros(attrs.size(0)).to(attrs.device)
            for weight, metric in zip(self.weights, self.cmpd_metrics):
                values += weight * metric.set_explainer(self.explainer).evaluate(inputs, targets, attrs)
            return values

    if args.model == "tab_resnet":
        expr = AutoExplanation(
            model=model,
            data=test_loader,
            modality=modality,
            target_input_keys=[0], # Current test_loader batches data as tuple of (inputs, targets). 0 means the location of inputs in the tuple
            target_class_extractor=lambda outputs: outputs.argmax(-1),
            label_key='labels',
            target_labels=False, # Gets attributions on the prediction for all explainer if False.
        )
    elif args.model == "xgb":
        from sklearn.cluster import KMeans as SklearnKMeans
        from pnpxai.explainers.utils.baselines import BaselineFunction
        from pnpxai.explainers.base import Tunable
        from pnpxai.explainers.types import TunableParameter


        class KMeans(BaselineFunction, Tunable):
            def __init__(self, background_data, n_clusters=8):
                self.background_data = background_data
                self.n_clusters = TunableParameter(
                    name='n_clusters',
                    current_value=n_clusters,
                    dtype=int,
                    is_leaf=True,
                    space={'low': 8, 'high': len(background_data)//10, 'step': 10},
                )
                self.kmeans_ = SklearnKMeans(n_clusters).fit(background_data)
                BaselineFunction.__init__(self)
                Tunable.__init__(self)
                self.register_tunable_params([self.n_clusters])

            def __call__(self, inputs):
                if inputs.ndim == 3:
                    inputs = inputs.squeeze(1)
                cluster_ids = self.kmeans_.predict(inputs.to(torch.float64).numpy())
                cluster_centers = self.kmeans_.cluster_centers_[cluster_ids]
                return torch.from_numpy(cluster_centers).to(inputs.device)
            
        expr = Experiment(
            model=model,
            data=test_loader,
            modality=modality,
            target_input_keys=[0],  # feature location in batch from dataloader
            target_class_extractor=lambda outputs: outputs.argmax(-1),  # extract target class from output batch
            label_key=-1,  # label location in input batch from dataloader
        )

        # add metrics
        expr.metrics.add('abpc', AbPC)

        # add explainers
        expr.explainers.add('kernel_shap', KernelShap)

        # remove unused baseline functions
        expr.modality.util_functions['baseline_fn'].delete('zeros')
        expr.modality.util_functions['baseline_fn'].delete('mean')

        # add new baseline functions
        expr.modality.util_functions['baseline_fn'].add('kmeans', KMeans)
        expr.modality.util_functions['baseline_fn'].add_default_kwargs(
            'background_data', X_test)
    else:
        raise ValueError("Invalid model")
    
    expr.metrics.add('cmpx', Complexity)
    expr.metrics.add('cmpd', CompoundMetric)

    metric_options = {}
    if args.metric == "cmpd":
        metric_options = {
            'cmpd_metrics': [
                expr.create_metric('abpc'), 
                expr.create_metric('cmpx'),
            ],
            'weights': [.7, -.3]
        }
    
    
    path = f"results/{dataset_nm}/{args.model}/{args.framework}/{args.explainer}"
    explanation = torch.tensor(np.load(f"{path}/explanations.npy"))
    metric = expr.create_metric(metric_key=args.metric, **metric_options)
    dummy_exp = expr.create_explainer(explainer_key="kernel_shap")
    
    res = []
    for i in range(explanation.shape[0]):
        inputs = {0: test_input[i].unsqueeze(0)}
        targets = model(inputs[0]).argmax(-1)
        exp = explanation[i].unsqueeze(0)
        evals = metric.set_explainer(dummy_exp).evaluate(
            inputs, targets, exp,
        )
        res.append(evals.item())
    res = np.array(res)
    np.save(f"{path}/{args.metric}.npy", res)
    logging.info(f"Saved {args.metric} results to {path}/{args.metric}.npy")



#     feature_metadata = pickle.load(open(f"data/{dataset_nm}/feature_metadata.pkl", "rb"))
#     raw_data = pd.read_csv(f"data/{dataset_nm}/raw_data.csv")
#     invert_input_array = functools.partial(_invert_input_array, feature_metadata=feature_metadata)
#     transform = functools.partial(_transform, feature_metadata=feature_metadata)

#     kmeans = KMeans(n_clusters=50, random_state=42)
#     kmeans.fit(X_train)

#     res = kmeans.predict(X_train)
#     _, counts = np.unique(res, return_counts=True)
#     weight = counts / counts.sum()

#     bg_data = find_closest_data_with_center(X_train, kmeans.cluster_centers_)
#     bg_data = invert_input_array(bg_data)
#     train_data = invert_input_array(X_train)

#     if args.model == "xgb":
#         model = xgb.XGBClassifier()
#         model.load_model(f"data/{dataset_nm}/xgb_model.json")

#     elif args.model == "tab_resnet":
#         model = TabResNet(X_train.shape[1], 2, num_blocks=1)
#         model.load_state_dict(torch.load(f"data/{dataset_nm}/resnet_model.pth"))
#         model.eval()

#     path = f"results/{dataset_nm}/{args.model}/{args.framework}/{args.explainer}"
#     explanation = np.load(f"{path}/explanations.npy")

#     record = []
#     check = [[],[]]
#     if args.metric == "abpc":
#         for i in range(explanation.shape[0]):
#             abpc, morf_curve, lerf_curve = compute_abpc(X_test[[i]], bg_data, model, transform, explanation[i], weight)
#             record.append(abpc)
#             check[0].append(morf_curve)
#             check[1].append(lerf_curve)
#     else:
#         raise ValueError("Invalid metric")

# np.save(f"{path}/{args.metric}.npy", np.array(record))
# np.save(f"{path}/morf.npy", np.array(check[0]))
# np.save(f"{path}/lerf.npy", np.array(check[1]))
# logging.info(f"Saved {args.metric} results to {path}/{args.metric}.npy")

# python script/evaluate.py --dataset "Wine Quality" --model tab_resnet --framework openxai --explainer lime --metric abpc 
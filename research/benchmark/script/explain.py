import os
import random
import functools
import argparse
import logging
import pickle
import yaml
from collections import defaultdict, deque

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import xgboost as xgb

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
        if np.isin('missing', X[[k]].values):
            X[[k]] = X[[k]].replace("missing", v['encoder'].categories_[0][-1])
        preprocessed = v['encoder'].transform(X[[k]].values)
        # preprocessed = v['encoder'].transform(X[[k]])
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


def validate_args(args):
    args.dataset = args.dataset.replace('\\','')
    if args.dataset not in ["Adult", "Bank Marketing", "Statlog (German Credit Data)", "Wine Quality"]:
        raise ValueError("Invalid dataset")
    
    if args.framework not in ["omnixai", "openxai", "pnpxai"]:
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
    parser.add_argument("--explainer", type=str)
    parser.add_argument("--batch_size", type=int, default=32)

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

    X_test = np.load(f"data/{dataset_nm}/X_test.npy")[:100]
    y_test = np.load(f"data/{dataset_nm}/y_test.npy")

    feature_metadata = pickle.load(open(f"data/{dataset_nm}/feature_metadata.pkl", "rb"))
    raw_data = pd.read_csv(f"data/{dataset_nm}/raw_data.csv")

    if args.model == "xgb":
        model = xgb.XGBClassifier()
        model.load_model(f"data/{dataset_nm}/xgb_model.json")

    elif args.model == "tab_resnet":
        model = TabResNet(X_train.shape[1], 2, num_blocks=1)
        model.load_state_dict(torch.load(f"data/{dataset_nm}/resnet_model.pth"))
        model.eval()

    if args.framework == "omnixai":
        from omnixai.data.tabular import Tabular
        from omnixai.explainers.tabular import TabularExplainer

        NAME_MAP = {
            "lime" : "LimeTabular",
            "shap" : "ShapTabular"
        }

        explainer_nm = NAME_MAP[args.explainer]

        raw_data = raw_data.fillna("missing")
        train_data = Tabular(raw_data, categorical_columns=[c for c in raw_data.columns if feature_metadata[c]["type"] == "categorical"])
        transform = functools.partial(_transform, feature_metadata=feature_metadata)
        invert_input_array = functools.partial(_invert_input_array, feature_metadata=feature_metadata)

        if args.model == "xgb":
            target_function = model.predict_proba

        def prep(z):
            return transform(z.data.fillna("missing"))

        explainer = TabularExplainer(
            explainers=[explainer_nm],
            mode="classification",                             # The task type
            data=train_data,
            model=target_function,
            preprocess=prep,
        )

        test_instances = invert_input_array(X_test).fillna("missing")
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

    elif args.framework == "openxai":
        from torch.utils.data import DataLoader, TensorDataset
        from openxai import Explainer
        from openxai.experiment_utils import fill_param_dict

        # 입력 데이터 및 모델 예측값 설정
        test_input = torch.tensor(X_test, dtype=torch.float32)
        explainer_name = args.explainer
        train_input = None
        explainer_params = {}

        # Explainer가 lime 또는 ig인 경우 학습 데이터 설정
        if explainer_name in ['lime', 'ig']:
            train_input = torch.tensor(X_train, dtype=torch.float32)
            explainer_params = fill_param_dict(explainer_name, {}, train_input)

        # Explainer 객체 생성
        explainer = Explainer(method=explainer_name, model=model, param_dict=explainer_params)

        # 모델 예측값 계산
        predicted_labels = model(test_input).detach().argmax(dim=1)

        # 데이터셋 및 DataLoader 생성
        dataset = TensorDataset(test_input, predicted_labels)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # 설명값 생성 및 저장
        all_explanations = []
        for batch_inputs, batch_labels in data_loader:
            batch_explanations = explainer.get_explanations(batch_inputs, label=batch_labels)
            all_explanations.append(batch_explanations)
            logger.info(f"Explaining batch... ({len(all_explanations)}/{len(data_loader)})")

        # 설명값 병합
        combined_explanations = torch.cat(all_explanations, dim=0)

        # Feature 별 설명값 변환
        processed_explanations = []
        for feature_name, feature_info in feature_metadata.items():
            if feature_info['type'] == 'categorical':
                feature_index = feature_info['index']
                onehot_encoded = test_input[:, feature_index]
                explanation_values = combined_explanations[:, feature_index]
                categorical_explanation = (onehot_encoded * explanation_values).sum(dim=1)
                processed_explanations.append(categorical_explanation)
            else:
                feature_index = feature_info['index']
                processed_explanations.append(combined_explanations[:, feature_index])

        # 최종 설명값 텐서로 변환
        explanations = torch.stack(processed_explanations, dim=1).detach().numpy()

    elif args.framework == "pnpxai":
        import pandas as pd
        from tqdm import tqdm
        import itertools
        from collections import defaultdict

        from torch.utils.data import DataLoader

        from pnpxai import Experiment, AutoExplanation
        from pnpxai.core.modality.modality import Modality
        from pnpxai.explainers import Lime, KernelShap
        from pnpxai.evaluator.metrics import AbPC, Metric, Complexity

        test_input = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
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

            expr.metrics.delete('morf')
            expr.metrics.delete('lerf')

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

            # add explainers
            expr.explainers.add('kernel_shap', KernelShap)
            expr.explainers.add('lime', Lime)

            # add metrics
            expr.metrics.add('abpc', AbPC)

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

        if args.explainer:
            PNP_NAME_MAP = {
                "grad" : "gradient",
                "ig" : "grad_x_input",
            }
            explanations = np.zeros_like(X_test)
            exp = PNP_NAME_MAP[args.explainer]
            met = "abpc"
            # met = "ab_p_c" if args.model != "xgb" else "abpc" # 이 부분 다시 확인
            for i in range(X_test.shape[0]):
                output = expr.run_batch(
                    explainer_key=exp,
                    metric_key=met,
                    data_ids=[i],
                )
                explanations[i] = output.explanations.detach().cpu()
        else:
            # optimize all
            with open('configs/optuna_config.yaml', 'r') as f:
                optuna_config = yaml.safe_load(f)

            records = []
            best_params = defaultdict(dict)
            combs = list(itertools.product(
                expr.explainers.choices,
                expr.metrics.choices,
            ))
            pbar = tqdm(combs, total=len(combs))
            for explainer_key, metric_key in pbar:
                # if expr.is_tunable(explainer_key): # skip if there's no tunable for an explainer
                metric_options = {}
                if metric_key == 'cmpd':
                    metric_options['cmpd_metrics'] = [
                        expr.create_metric('abpc'),
                        expr.create_metric('cmpx'),
                    ]
                    metric_options['weights'] = [.8, -.2]
                pbar.set_description(f'Optimizing {explainer_key} on {metric_key}')
                direction = {
                    'abpc': 'maximize',
                    'cmpx': 'minimize',
                    'cmpd': 'maximize',
                }.get(metric_key)

                disable_tunable_params = {}
                if explainer_key in ['lime', 'kernel_shap']:
                    disable_tunable_params['n_samples'] = 2048

                opt_results = expr.optimize(
                    explainer_key=explainer_key,
                    metric_key=metric_key,
                    metric_options=metric_options,
                    direction=direction,
                    disable_tunable_params=disable_tunable_params,
                    **optuna_config
                )
                records.append({
                    'explainer': explainer_key,
                    'metric': metric_key,
                    'value': opt_results.study.best_trial.value,
                })
                best_params[explainer_key][metric_key] = opt_results.study.best_params

                PNP_INV_MAP = {
                    "kernel_shap" : "shap",
                    "lime" : "lime",
                    "gradient" : "grad",
                    "grad_x_input" : "ig",
                    "integrated_gradients" : "itg",
                    "smooth_grad" : "sg",
                    "lrp_uniform_epsilon" : "lrp",
                    "var_grad" : "vg",
                }

                explanations = np.zeros_like(X_test)
                opt_explainer = opt_results.explainer
                targets = model(test_input.squeeze()).argmax(-1)
                if explainer_key in ["lime", "kernel_shap"]:
                    _test_input = test_input.squeeze(1)
                else:
                    _test_input = test_input
                explanations = opt_explainer.attribute(_test_input, targets)[0].detach().cpu().numpy()
                exp_name = PNP_INV_MAP[explainer_key]
                path = f"results/{dataset_nm}/{args.model}/{args.framework}/{exp_name}"
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(f"{path}/explanations.npy", explanations)

            df = pd.DataFrame.from_records(records)
            path = f"results/{dataset_nm}/{args.model}/{args.framework}"
            if not os.path.exists(path):
                os.makedirs(path)
            df.to_csv(path + "/evaluation.csv", index=False)
        

        # tmp_exp = np.zeros_like(X_test)
        # exp = PNP_NAME_MAP[args.explainer]
        # met = "abpc"
        # # met = "ab_p_c" if args.model != "xgb" else "abpc" # 이 부분 다시 확인
        # for i in range(X_test.shape[0]):
        #     output = expr.run_batch(
        #         explainer_key=exp,
        #         metric_key=met,
        #         data_ids=[i],
        #     )
        #     tmp_exp[i] = output.explanations.detach().cpu()


        # # Feature 별 설명값 변환
        # explanations = np.zeros((X_test.shape[0], len(feature_metadata)))
        # for i, (feature_name, feature_info) in enumerate(feature_metadata.items()):
        #     feature_index = feature_info['index']
        #     if feature_info['type'] == 'categorical':
        #         onehot_encoded = test_input[:, 0, feature_index]
        #         explanation_values = tmp_exp[:, feature_index]
        #         categorical_explanation = (onehot_encoded * explanation_values).sum(dim=1)
        #         explanations[:, i] = categorical_explanation
        #     else:
        #         explanations[:, i] = tmp_exp[:, feature_index]

# save explanation
if args.explainer:
    path = f"results/{dataset_nm}/{args.model}/{args.framework}/{args.explainer}"
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(f"{path}/explanations.npy", explanations)

# python script/explain.py --dataset "Bank Marketing" --model xgb --framework omnixai --explainer lime
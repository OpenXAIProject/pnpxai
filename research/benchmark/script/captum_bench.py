import os
import argparse, pickle, random, yaml, logging
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
from tqdm import tqdm
from captum.attr import (
    Lime, IntegratedGradients, Saliency, InputXGradient, NoiseTunnel, LRP
)
from captum.attr import KernelShap as CaptumKernelShap
from captum.attr._utils.lrp_rules import EpsilonRule

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans as SklearnKMeans

from pnpxai import Experiment, AutoExplanation
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers import KernelShap
from pnpxai.evaluator.metrics import AbPC, Metric, Complexity, Sensitivity
from pnpxai.explainers.utils.baselines import BaselineFunction
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter
# ---------------- utils ----------------

# class TorchModelForXGBoost(nn.Module):
#     """Captum-friendly wrapper around XGBoost predict_proba."""
#     def __init__(self, xgb_model):
#         super().__init__()
#         self.xgb_model = xgb_model
#         self._dummy = nn.Linear(1, 1)  # Dummy layer for device handling

#     def forward(self, x):
#         if x.ndim == 3:
#             x = x.squeeze(0)
#         x_np = x.detach().cpu().numpy()
#         out = self.xgb_model.predict_proba(x_np)
#         out_tensor = torch.from_numpy(out).to(x.device)  # (N, 2)
#         return out_tensor[:, 1]  # (N,)
    
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


def load_data(dataset_nm):
    X_train = np.load(f"data/{dataset_nm}/X_train.npy")
    y_train = np.load(f"data/{dataset_nm}/y_train.npy")
    X_test  = np.load(f"data/{dataset_nm}/X_test.npy")
    y_test  = np.load(f"data/{dataset_nm}/y_test.npy")
    feature_metadata = pickle.load(open(f"data/{dataset_nm}/feature_metadata.pkl","rb"))
    return X_train, y_train, X_test, y_test, feature_metadata

# --------------- main ------------------
dataset_nm="Wine Quality"
seed=42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
os.makedirs("results", exist_ok=True)

# 1. load data
X_tr, y_tr, X_te, y_te, _ = load_data(dataset_nm)
X_te_t = torch.tensor(X_te, dtype=torch.float32)

# 2. load models
models_cfg = {
    "xgb": {
        "model": TorchModelForXGBoost(xgb.XGBClassifier()),
        "attr": ["lime", "kernelshap"]
        # "attr": ["kernelshap","lime"]
    },
    "tab_resnet": {
        "model": None,   # load below
        "attr": ["gradient","gradxinput","ig","smoothgrad","lrp"]
    },
}
# XGB load
xgb_raw = xgb.XGBClassifier()
xgb_raw.load_model(f"data/{dataset_nm}/xgb_model.json")
models_cfg["xgb"]["model"].xgb_model = xgb_raw

# TabResNet load
from models.tab_resnet import TabResNet
resnet = TabResNet(X_tr.shape[1], 2, num_blocks=1)
resnet.load_state_dict(torch.load(f"data/{dataset_nm}/resnet_model.pth"))
resnet.eval()
models_cfg["tab_resnet"]["model"] = resnet

# 3. explainer factory
def make_explainer(tag, model):
    if tag=="kernelshap": return CaptumKernelShap(model)
    if tag=="lime":       return Lime(model, interpretable_model=None)
    if tag=="gradient":   return Saliency(model)
    if tag=="gradxinput": return InputXGradient(model)
    if tag=="ig":         return IntegratedGradients(model, multiply_by_inputs=True)
    if tag=="smoothgrad": return NoiseTunnel(Saliency(model))          # default stdev, nt_type='smoothgrad'
    if tag=="lrp":        
        model.res_blocks[0].bn.rule = EpsilonRule()
        model.bn.rule = EpsilonRule()
        return LRP(model)                       # for simple feed-forward nets
        # return LayerLRP(model, layer=model.embedding)                       # for simple feed-forward nets
    raise ValueError(tag)

# 4. loop over models / explainers
nsamples = 2048
results = {}
for model_key, cfg in models_cfg.items():
    mdl = cfg["model"]
    mdl.to("cpu")
    results[model_key] = {}
    for tag in cfg["attr"]:
        print(f"[{model_key}] calculating {tag} …")
        explainer = make_explainer(tag, mdl)
        target = mdl(X_te_t).argmax(dim=1)
        # target = mdl(X_te_t).argmax(dim=1)  if model_key=="tab_resnet" else mdl(X_te_t) > 0.5
        if tag=="ig":     
            attrs = explainer.attribute(X_te_t, target=target, n_steps=50).detach().numpy()
        elif tag=="smoothgrad":
            attrs = explainer.attribute(X_te_t, target=target, nt_samples=50, nt_type='smoothgrad').detach().numpy()
        elif tag in ("kernelshap","lime"):
            attrs_list = []
            for i in tqdm(range(len(X_te_t))):
                baseline = torch.tensor(X_tr[np.random.choice(len(X_tr), 1)], dtype=torch.float32)  # (1, feature_dim)
                input_i = X_te_t[i].unsqueeze(0)  # (1, feature_dim)
                attr_i = explainer.attribute(input_i, target=target[i], baselines=baseline, n_samples=nsamples)
                attrs_list.append(attr_i.detach().cpu().numpy())

            attrs = np.concatenate(attrs_list, axis=0)  # (batch_size, feature_dim)

        else:             
            attrs = explainer.attribute(X_te_t, target=target).detach().numpy()

        results[model_key][tag] = attrs
        # 저장 npy (필요하면 변경)
        np.save(f"results/{dataset_nm.replace(' ','_')}_{model_key}_{tag}.npy", attrs)

# 5. 요약 yaml (구조만 기록)
yaml.safe_dump(
    {k:list(v.keys()) for k,v in results.items()},
    open(f"results/summary_{dataset_nm.replace(' ','_')}.yml","w")
)
print("Done!")


total_res = []
test_input = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1)
test_loader = DataLoader(
    test_input,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
)

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
    
for model_key, cfg in models_cfg.items():
    model = cfg["model"]
    if model_key == "tab_resnet":
        expr = AutoExplanation(
            model=model,
            data=test_loader,
            modality=modality,
            target_input_keys=[0], # Current test_loader batches data as tuple of (inputs, targets). 0 means the location of inputs in the tuple
            target_class_extractor=lambda outputs: outputs.argmax(-1),
            label_key='labels',
            target_labels=False, # Gets attributions on the prediction for all explainer if False.
        )
    elif model_key == "xgb":
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
                return torch.from_numpy(cluster_centers).to(torch.float32).to(inputs.device)
            
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
            'background_data', X_te)
    else:
        raise ValueError("Invalid model")

    expr.metrics.add('cmpx', Complexity)
    expr.metrics.add('cmpd', CompoundMetric)
    expr.metrics.add('sens', Sensitivity)

    for tag in cfg["attr"]:
        for metric in ['abpc', 'cmpx', 'sens', 'cmpd']:
            print(f"[{model_key}] calculating {tag} {metric} …")
            metric_options = {}
            if metric == "cmpd":
                metric_options = {
                    'cmpd_metrics': [
                        expr.create_metric('abpc'), 
                        expr.create_metric('cmpx'),
                        expr.create_metric('sens'),
                    ],
                    'weights': [.7, -.2, -.1]
                }

            explanation = np.load(f"results/{dataset_nm.replace(' ','_')}_{model_key}_{tag}.npy")
            explanation = torch.tensor(explanation, dtype=torch.float32)
            metric = expr.create_metric(metric_key=metric, **metric_options)
            dummy_exp = expr.create_explainer(explainer_key="kernel_shap")

            res = []
            for i in range(explanation.shape[0]):
                inputs = {0: test_input[i].unsqueeze(0)}
                targets = model(test_input[i]).argmax(-1)
                exp = explanation[i].unsqueeze(0)
                    
                evals = metric.set_explainer(dummy_exp).evaluate(
                    inputs, targets, exp,
                )
                res.append(evals.item())
            res = np.array(res)
            total_res.append({
                "model": model_key,
                "tag": tag,
                "metric": metric,
                "res": res
            })
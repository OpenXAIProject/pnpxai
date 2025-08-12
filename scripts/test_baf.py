'''
This script implements benchmark test on various explainers and gets the best
performing explainer on BAF (bank account fraud detection) task, using PnPXAI
framework.

Prerequisites:
- This script makes `--data_dir` and downloads baf data from kaggle, if it does
  not exist. Please install kaggle e.g.

  ```bash
  pip install kaggle
  ```

Flags:
--fast_dev_run: runs the script with small samples and trials

Example:

```bash
python -m scripts.test_baf --model tab_resnet --data_dir baf --log_dir baf --fast_dev_run
```
'''

import argparse
import os
import re
import itertools
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost

from pnpxai import XaiRecommender, Experiment, AutoExplanation
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers import Lime, KernelShap
from pnpxai.evaluator.metrics import AbPC, MoRF, LeRF


BAF_MODEL_CHOICES = ['tab_resnet', 'xgb']


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=BAF_MODEL_CHOICES, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--fast_dev_run', action='store_true')


#------------------------------------------------------------------------------#
#----------------------------------- data -------------------------------------#
#------------------------------------------------------------------------------#

class PandasDataset(Dataset):
    def __init__(self, inputs: pd.DataFrame, labels: pd.Series):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs.iloc[idx], self.labels.iloc[idx]


def collate_fn(batch):
    inputs = torch.stack([torch.from_numpy(d[0].values) for d in batch]).to(torch.float)
    labels = torch.tensor([d[1] for d in batch], dtype=torch.long)
    return inputs, labels


def download_data(root_dir):
    ZIPFILE_NAME = 'bank-account-fraud-dataset-neurips-2022'
    FILE_NAME = 'Base.csv'
    raw_dir = os.path.join(root_dir, 'raw')
    zipfile_path = os.path.join(raw_dir, ZIPFILE_NAME)
    file_path = os.path.join(raw_dir, FILE_NAME)
    if not os.path.exists(file_path):
        print("Downloading the dataset...")
        os.makedirs(raw_dir, exist_ok=True)
        os.system(f"kaggle datasets download -d sgpjesus/{ZIPFILE_NAME} -p {raw_dir}")
        os.system(f"unzip {zipfile_path} -d {raw_dir}")
    return file_path


def preprocess_data(file_path, mul=2, random_state=42):
    df = pd.read_csv(file_path)
    is_train = df['month'] < 5
    is_valid = (df['month'] >= 5) & (df['month'] < 6)
    is_test = df['month'] >= 6
    is_fraud = df['fraud_bool'] == 1

    balance_df = lambda is_data: pd.concat([
        df.loc[is_data&is_fraud],
        df.loc[is_data&(~is_fraud)].sample((is_data&is_fraud).sum()*mul, random_state=random_state)
    ]).reset_index(drop=True).drop(columns=['month'])
    dfs = {'train': balance_df(is_train), 'valid': balance_df(is_valid), 'test': balance_df(is_test)}

    scaler = StandardScaler()
    ohe = OneHotEncoder()
    preprocessed = {}
    for split, features in dfs.items():
        preprocessed[f'y_{split}'] = features.pop('fraud_bool')

        float_cols = features.select_dtypes(include=[float, int]).columns
        float_features = pd.concat([features.pop(c) for c in float_cols], axis=1)
        float_features_data = scaler.fit_transform(float_features) if split == 'train' else \
                scaler.transform(float_features)
        float_features = pd.DataFrame(
            data=float_features_data,
            index=float_features.index,
            columns=float_features.columns,
        )

        categ_cols = features.select_dtypes(include=['object', int]).columns
        categ_features = pd.concat([features.pop(c) for c in categ_cols], axis=1)
        categ_features_data = ohe.fit_transform(categ_features) if split == 'train' else \
            ohe.transform(categ_features)
        categ_features = pd.DataFrame(
            data=categ_features_data.toarray(),
            index=categ_features.index,
            columns=[f'{c}_{v}' for c, values in zip(categ_cols, ohe.categories_) for v in values]
        )
        preprocessed[f'x_{split}'] = pd.concat([float_features, categ_features], axis=1)
    return preprocessed


#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

# tab resnet
class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResNetBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        y = torch.relu(self.fc1(self.bn(x)))
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return torch.add(x, y)

    
class TabResNet(nn.Module):
    def __init__(self, in_features, out_features, num_blocks=1, embedding_dim=128):
        super(TabResNet, self).__init__()
        self.embedding = nn.Linear(in_features, embedding_dim)
        self.res_blocks = []
        for i in range(num_blocks):
            self.res_blocks.append(ResNetBlock(embedding_dim, embedding_dim))
        self.res_blocks = nn.ModuleList(self.res_blocks)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, out_features)
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.res_blocks:
            x = block(x)
        x = torch.relu(self.bn(x))
        x = self.fc(x)
        return x


def train(
    model_dir,
    checkpoint_nm,
    model,
    train_loader,
    valid_inputs,
    valid_labels,
    device,
):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, checkpoint_nm)
    if os.path.exists(model_path):
        return model_path
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    # Train the model
    for epoch in range(10):
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader)}")

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            y_pred = model(valid_inputs)
            loss = loss_fn(y_pred, valid_labels)
            valid_loss = loss.item()

            print(f"Validation Loss: {valid_loss}")
            y_pred = torch.argmax(y_pred, dim=1)
            print(classification_report(valid_labels.to('cpu'), y_pred.to('cpu')))
    torch.save(model.state_dict(), model_path)
    return model_path


class TorchModelForXGBoost(nn.Module):
    def __init__(self, xgb_model):
        super().__init__()
        self.xgb_model = xgb_model
        self._dummy_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor):
        out = self.xgb_model.predict_proba(x.cpu().numpy())
        return torch.from_numpy(out)



#------------------------------------------------------------------------------#
#----------------------------------- kmeans -----------------------------------#
#------------------------------------------------------------------------------#

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
        cluster_ids = self.kmeans_.predict(inputs.numpy())
        cluster_centers = self.kmeans_.cluster_centers_[cluster_ids]
        return torch.from_numpy(cluster_centers).to(inputs.device)


#------------------------------------------------------------------------------#
#----------------------------------- main -------------------------------------#
#------------------------------------------------------------------------------#

def main_resnet(args):
    # setup
    use_gpu = torch.cuda.is_available() and not args.disable_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')
    torch.manual_seed(args.seed)

    # prepare data
    data_fpth = download_data(args.data_dir) # download data and get the filepath
    data = preprocess_data(data_fpth) # load and preprocess data

    train_set = PandasDataset(data['x_train'], data['y_train'])
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=use_gpu,
    )
    valid_set = PandasDataset(data['x_valid'], data['y_valid'])
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=use_gpu,
    )
    test_set = PandasDataset(data['x_test'], data['y_test'])
    if args.fast_dev_run:
        indices = list(range(args.batch_size*2))
        test_set = Subset(test_set, indices=indices)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=use_gpu,
    )
    valid_inputs, valid_labels = next(iter(valid_loader)) # use small sample for validation
    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)


    # prepare model
    model = TabResNet(
        in_features=len(data['x_train'].columns),
        out_features=len(data['y_train'].unique()),
    )
    ckpt_fpth = train(
        model_dir=args.log_dir,
        checkpoint_nm='tabresnet.pkl',
        model=model,
        train_loader=train_loader,
        valid_inputs=valid_inputs,
        valid_labels=valid_labels,
        device=device,
    )
    model.load_state_dict(torch.load(ckpt_fpth))
    model.to(device).eval()

    # prepare modality
    sample_batch = next(iter(test_loader))
    modality = Modality(
        dtype=sample_batch[0].dtype,
        ndims=sample_batch[0].dim(),
    )

    '''
    #--------------------------------------------------------------------------#
    #------------------------------- recommend --------------------------------#
    #--------------------------------------------------------------------------#

    # You can get pnpxai recommendation results without AutoExplanation as followings:

    recommended = XaiRecommender().recommend(
        modality=modality,
        model=model,
    )
    
    recommended.print_tabular()
    '''

    '''
    #--------------------------------------------------------------------------#
    #------------------------------ experiment --------------------------------#
    #--------------------------------------------------------------------------#

    # You can manually create experiment as followings:
    expr = Experiment(
        model=model,
        data=test_loader,
        modality=modality,
        target_input_keys=[0],  # feature location in batch from dataloader
        target_class_extractor=lambda outputs: outputs.argmax(-1),  # extract target class from output batch
        label_key=-1,  # label location in input batch from dataloader
    )

    # add recommended explainers recommended
    camel_to_snake = lambda name: re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    for explainer_type in recommended.explainers:
        name = camel_to_snake(explainer_type.__name__)
        expr.explainers.add(key=name, value=explainer_type)

    # add a metric
    expr.metrics.add(key='abpc', value=AbPC)
    '''


    #--------------------------------------------------------------------------#
    #--------------------------- auto explanation -----------------------------#
    #--------------------------------------------------------------------------#

    # create experiment using auto explanation
    expr = AutoExplanation(
        model=model,
        data=test_loader,
        modality=modality,
        target_input_keys=[0], # Current test_loader batches data as tuple of (inputs, targets). 0 means the location of inputs in the tuple
        target_class_extractor=lambda outputs: outputs.argmax(-1),
        label_key='labels',
        target_labels=False, # Gets attributions on the prediction for all explainer if False.
    )
    

    # You can browse available explainer_keys and metric_keys as followings:
    print(expr.explainers.choices)
    print(expr.metrics.choices)

    # optimize all
    records = []
    best_params = defaultdict(dict)
    combs = list(itertools.product(
        expr.explainers.choices,
        expr.metrics.choices,
    ))
    pbar = tqdm(combs, total=len(combs))
    for explainer_key, metric_key in pbar:
        if expr.is_tunable(explainer_key): # skip if there's no tunable for an explainer
            pbar.set_description(f'Optimizing {explainer_key} on {metric_key}')
            direction = {
                'morf': 'minimize',
                'lerf': 'maximize',
                'abpc': 'maximize',
            }.get(metric_key)
            disable_tunable_params = {}
            if explainer_key in ['lime', 'kernel_shap']:
                disable_tunable_params['n_samples'] = 100
            opt_results = expr.optimize(
                explainer_key=explainer_key,
                metric_key=metric_key,
                direction=direction,
                disable_tunable_params=disable_tunable_params,
                sampler='random',
                seed=42,
                num_threads=16,
                show_progress=not args.fast_dev_run,
                n_trials=2 if args.fast_dev_run else 100,
            )
            records.append({
                'explainer': explainer_key,
                'metric': f'Best {metric_key}',
                'value': opt_results.study.best_trial.value,
            })
            best_params[explainer_key][metric_key] = opt_results.study.best_params
    df = pd.DataFrame.from_records(records)
    summary_table = df.set_index(
        ['explainer', 'metric'])['value'].unstack('metric')
    print('-------- Summary --------')
    print(summary_table)
    print('------ Best Params ------')
    pprint(best_params)


def main_xgb(args):
    # data
    data_fpth = download_data(args.data_dir) # download data and get the filepath
    data = preprocess_data(data_fpth) # load and preprocess data

    test_set = PandasDataset(data['x_test'], data['y_test'])
    if args.fast_dev_run:
        indices = list(range(args.batch_size*2))
        test_set = Subset(test_set, indices=indices)
    test_loader = DataLoader(
        test_set,
        batch_size=len(test_set),
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    # model
    xgb_model = xgboost.XGBClassifier()
    xgb_model.fit(data['x_train'], data['y_train'])
    torch_model = TorchModelForXGBoost(xgb_model=xgb_model)

    # modality
    sample_batch = next(iter(test_loader))
    modality = Modality(
        dtype=sample_batch[0].dtype,
        ndims=sample_batch[0].dim(),
    )

    #--------------------------------------------------------------------------#
    #------------------------------ experiment --------------------------------#
    #--------------------------------------------------------------------------#

    # You can manually create experiment as followings:
    expr = Experiment(
        model=torch_model,
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
    expr.metrics.add('morf', MoRF)
    expr.metrics.add('lerf', LeRF)

    # remove unused baseline functions
    expr.modality.util_functions['baseline_fn'].delete('zeros')
    expr.modality.util_functions['baseline_fn'].delete('mean')

    # add new baseline functions
    expr.modality.util_functions['baseline_fn'].add('kmeans', KMeans)
    expr.modality.util_functions['baseline_fn'].add_default_kwargs(
        'background_data', test_set.inputs.astype('float32').to_numpy())

    records = []
    best_params = defaultdict(dict)
    combs = list(itertools.product(
        expr.explainers.choices,
        expr.metrics.choices,
    ))
    pbar = tqdm(combs, total=len(combs))
    for explainer_key, metric_key in pbar:
        if expr.is_tunable(explainer_key): # skip if there's no tunable for an explainer
            pbar.set_description(f'Optimizing {explainer_key} on {metric_key}')
            direction = {
                'morf': 'minimize',
                'lerf': 'maximize',
                'abpc': 'maximize',
            }.get(metric_key)

            opt_results = expr.optimize(
                explainer_key=explainer_key,
                metric_key=metric_key,
                metric_options={'n_steps': sample_batch[0].size(1)}, # pixel flip feature by feature
                direction=direction,
                disable_tunable_params={'n_samples': 30}, # fix n_samples
                sampler='random',
                seed=42,
                num_threads=16,
                show_progress=not args.fast_dev_run,
                n_trials=2 if args.fast_dev_run else 100,
            )
            records.append({
                'explainer': explainer_key,
                'metric': f'Best {metric_key}',
                'value': opt_results.study.best_trial.value,
            })
            best_params[explainer_key][metric_key] = opt_results.study.best_params
    df = pd.DataFrame.from_records(records)
    summary_table = df.set_index(
        ['explainer', 'metric'])['value'].unstack('metric')
    print('-------- Summary --------')
    print(summary_table)
    print('------ Best Params ------')
    pprint(best_params)



if __name__ == '__main__':
    args = parser.parse_args()
    if args.model == 'tab_resnet':
        main_resnet(args)
    elif args.model == 'xgb':
        main_xgb(args)


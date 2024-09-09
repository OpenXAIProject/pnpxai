import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd


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
        os.system(f"rm {zipfile_path}")
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


ROOT_DIR = '/data1/geonhyeong/data/baf'
file_path = download_data(ROOT_DIR)
data = preprocess_data(file_path)

train_set = PandasDataset(data['x_train'], data['y_train'])
train_loader = DataLoader(train_set, batch_size=128, collate_fn=collate_fn, shuffle=True)
valid_set = PandasDataset(data['x_valid'], data['y_valid'])
valid_loader = DataLoader(valid_set, batch_size=1024, collate_fn=collate_fn, shuffle=True)
test_set = PandasDataset(data['x_test'], data['y_test'])
test_loader = DataLoader(test_set, batch_size=1024, collate_fn=collate_fn, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valid_inputs, valid_labels = next(iter(valid_loader)) # validate with a tensor
valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)

#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

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


def train(model_dir, checkpoint_nm, model, train_loader, valid_inputs, valid_labels):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, checkpoint_nm)
    if os.path.exists(model_path):
        return model_path
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

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


model = TabResNet(in_features=len(data['x_train'].columns), out_features=len(data['y_train'].unique()))
model = model.to(device)
model_dir = os.path.join(ROOT_DIR, 'models')
model_path = train(model_dir, 'tabresnet.pkl', model, train_loader, valid_inputs, valid_labels)
model.load_state_dict(torch.load(model_path))
model.eval()


#------------------------------------------------------------------------------#
#------------------------------- custom pnpxai --------------------------------#
#------------------------------------------------------------------------------#

from pnpxai import AutoExplanation
from pnpxai.core.modality import Modality, TextModality
from pnpxai.explainers.utils.function_selectors import FunctionSelector
from pnpxai.explainers.utils.baselines import BaselineFunction
from pnpxai.explainers.utils.feature_masks import NoMask1d
from pnpxai.explainers.utils.postprocess import (
    PostProcessor,
    Identity,
    MinMax,
    minmax,
)

from sklearn.cluster import KMeans as SklearnKMeans

class KMeans(BaselineFunction):
    def __init__(self, background_data, k=100, random_state=42, n_init='auto'):
        self.background_data = background_data
        self._kmeans = SklearnKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        ).fit(background_data.values)

    def __call__(self, inputs):
        clusters = self._kmeans.predict(inputs.cpu())
        centers = torch.Tensor(self._kmeans.cluster_centers_[clusters])
        return centers.to(inputs.device)

    def get_tunables(self):
        return {'k': (int, {'low': 10, 'high': 1000, 'step': 10})}


baseline_functions_for_tabular = {
    'kmean': KMeans,
}

feature_mask_functions_for_tabular = {
    'no_mask_1d': NoMask1d,
}

pooling_functions_for_tabular = {'identity': Identity}
normalization_functions_for_tabular = {'identity': Identity}


class TabularModality(Modality):
    EXPLAINERS = TextModality.EXPLAINERS

    def __init__(self, channel_dim, background_data):
        super(TabularModality, self).__init__(
            channel_dim,
            baseline_fn_selector=FunctionSelector(
                data=baseline_functions_for_tabular,
                default_kwargs={'background_data': background_data},
            ),
            feature_mask_fn_selector=FunctionSelector(feature_mask_functions_for_tabular),
            pooling_fn_selector=FunctionSelector(
                data=pooling_functions_for_tabular,
                default_kwargs={'channel_dim': channel_dim},
            ),
            normalization_fn_selector=FunctionSelector(normalization_functions_for_tabular),
        )
        self.background_data = background_data

    def get_default_baseline_fn(self):
        return self.baseline_fn_selector.select('kmean', k=100)

    def get_default_feature_mask_fn(self):
        return self.feature_mask_fn_selector.select('no_mask_1d')

    def get_default_postprocessors(self):
        return [
            PostProcessor(
                pooling_fn=self.pooling_fn_selector.select(pm),
                normalization_fn=self.normalization_fn_selector.select(nm),
            ) for pm in self.pooling_fn_selector.choices
            for nm in self.normalization_fn_selector.choices
        ]


class AutoExplanationForTabularClassification(AutoExplanation):
    def __init__(
        self,
        model,
        data,
        background_data,
        input_extractor,
        label_extractor,
        target_extractor,
        target_labels=False,
        channel_dim=-1,
    ):
        super().__init__(
            model=model,
            data=data,
            modality=TabularModality(channel_dim, background_data),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
        )


expr = AutoExplanationForTabularClassification(
    model=model.to(device),
    data=test_loader,
    background_data=data['x_test'],
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
)

# get all explanations
attrs_all = [{
    'explainer': explainer,
    'results': expr.run_batch(
        data_ids=range(4),
        explainer_id=explainer_id,
        postprocessor_id=0,
        metric_id=1,
    ),
} for explainer_id, explainer in enumerate(expr.manager.explainers)]


#------------------------------------------------------------------------------#
#------------------------------- optimization ---------------------------------#
#------------------------------------------------------------------------------#

data_id = 0
explainer_id = 3 # Lime
metric_id = 1 # ABPC

optimized = expr.optimize(
    data_ids=[data_id],
    explainer_id=explainer_id,
    metric_id=metric_id,
    direction='maximize', # less is better
    sampler='random', # Literal['tpe','random']
    n_trials=50, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
    seed=42, # seed for sampler: by default, None
)

print('Best/Explainer:', optimized.explainer) # get the optimized explainer
print('Best/PostProcessor:', optimized.postprocessor) # get the optimized postprocessor
print('Best/value:', optimized.study.best_trial.value) # get the optimized value

# Every trial in study has its explainer and postprocessor in user attr.
i = 25
print(f'{i}th Trial/Explainer', optimized.study.trials[i].user_attrs['explainer']) # get the explainer of i-th trial
print(f'{i}th Trial/PostProcessor', optimized.study.trials[i].user_attrs['postprocessor']) # get the postprocessor of i-th trial
print(f'{i}th Trial/value', optimized.study.trials[i].value)

# For example, you can use optuna's API to get the explainer and postprocessor of the worst trial
def get_worst_trial(study):
    valid_trials = [trial for trial in study.trials if trial.value is not None]
    return sorted(valid_trials, key=lambda trial: trial.value)[0]

worst_trial = get_worst_trial(optimized.study)
print('Worst/Explainer:', worst_trial.user_attrs['explainer'])
print('Worst/PostProcessor', worst_trial.user_attrs['postprocessor'])
print('Worst/value', worst_trial.value)


# ------------------------------------------------------------------------------#
# ------------------------------- visualization --------------------------------#
# ------------------------------------------------------------------------------#

import matplotlib.pyplot as plt

batch = expr.manager.batch_data_by_ids([data_id])
inputs = expr.input_extractor(batch)
labels = expr.label_extractor(batch)

# prepare trials to visualize
sorted_trials = sorted(
    [trial for trial in optimized.study.trials if trial.value is not None],
    key=lambda trial: trial.value,
)
trials_to_vis = {
    'Worst': sorted_trials[0],
    'Median': sorted_trials[len(sorted_trials)//2],
    'Best': sorted_trials[-1],
}

ylabs = [
    f'{nm}={"{:.2f}".format(v)}'
    for nm, v in zip(data['x_train'].columns, inputs[0])
]
xlabs = trials_to_vis.keys()
heatmaps = []
for trial in trials_to_vis.values():
    attrs = trial.user_attrs['explainer'].attribute(inputs, labels)
    postprocessed = trial.user_attrs['postprocessor'](attrs)
    heatmaps.append(postprocessed)
heatmaps = torch.cat(heatmaps).transpose(1, 0).detach().cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
hm = ax.imshow(heatmaps, cmap='viridis')
ax.set_xticks(range(len(xlabs)), labels=xlabs)
ax.set_yticks(range(len(ylabs)), labels=ylabs)
ax.set_aspect(len(xlabs)*1.5/len(ylabs))
fig.colorbar(hm, ax=ax, location='right')
ax.set_title(f'is_fraud={bool(labels[0])}: {expr.manager.get_explainer_by_id(explainer_id).__class__.__name__}')
fig.savefig('auto_explanation_baf_example.png')

import numpy as np


seed = 42
np.random.seed(seed)

#------------------------------------------------------------------------------#
#----------------------------------- data -------------------------------------#
#------------------------------------------------------------------------------#

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_breast_cancer


# classification
cls_data = load_breast_cancer(as_frame=True)
cls_x_train, cls_x_test, cls_y_train, cls_y_test = train_test_split(cls_data.data, cls_data.target, train_size=0.80)


# regression
reg_data = fetch_california_housing(as_frame=True)
reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(reg_data.data, reg_data.target, train_size=0.80)


#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

from xgboost import XGBClassifier, XGBRegressor


classifier = XGBClassifier()
classifier.fit(cls_x_train, cls_y_train)

regressor = XGBRegressor()
regressor.fit(reg_x_train, reg_y_train)


#------------------------------------------------------------------------------#
#---------------------------------- explain -----------------------------------#
#------------------------------------------------------------------------------#

from pnpxai.explainers import TabKernelShap, TabLime

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
    inputs = pd.concat([d[0] for d in batch], axis=1).T
    labels = pd.Series(data=[d[1] for d in batch], index=inputs.index)
    return inputs, labels


cls_dataset = PandasDataset(inputs=cls_x_test, labels=cls_y_test)
cls_loader = DataLoader(cls_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

reg_dataset = PandasDataset(inputs=reg_x_test, labels=reg_y_test)
reg_loader = DataLoader(reg_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

input_extractor = lambda batch: batch[0]
label_extractor = lambda batch: batch[1]


# classifier
cls_batch = next(iter(cls_loader))
cls_inputs = input_extractor(cls_batch)
cls_labels = label_extractor(cls_batch)
cls_outputs = classifier.predict_proba(cls_inputs)
cls_target_extractor = lambda outputs: outputs.argmax(-1)
cls_targets = cls_target_extractor(cls_outputs)

# classifier x lime
categorical_features = None
explainer = TabLime(
    model=classifier,
    background_data=cls_x_train,
    mode='classification',
)
cls_lime_attrs = explainer.attribute(cls_inputs, cls_targets)

# classifier x ks
explainer = TabKernelShap(
    model=classifier,
    background_data=cls_x_train,
    k_means=100,
    mode="classification"
)
cls_ks_attrs = explainer.attribute(cls_inputs, cls_targets)

reg_batch = next(iter(reg_loader))
reg_inputs = input_extractor(reg_batch)
reg_labels = label_extractor(reg_batch)
reg_outputs = regressor.predict(reg_inputs)
reg_target_extractor = lambda outputs: outputs
reg_targets = reg_target_extractor(reg_outputs)

# regressor x lime
explainer = TabLime(
    model=regressor,
    background_data=reg_x_train,
    mode='regression',
)
reg_lime_attrs = explainer.attribute(reg_inputs, reg_targets)

# regressor x ks
explainer = TabKernelShap(
    model=regressor,
    background_data=reg_x_train,
    k_means=100,
    mode='regression',
)
reg_ks_attrs = explainer.attribute(reg_inputs, reg_targets)


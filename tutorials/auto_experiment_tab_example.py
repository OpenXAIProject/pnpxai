import numpy as np


seed = 42
np.random.seed(seed)

#------------------------------------------------------------------------------#
#----------------------------------- data -------------------------------------#
#------------------------------------------------------------------------------#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from torch.utils.data import Dataset, DataLoader


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


data = load_breast_cancer(as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.80)
dataset = PandasDataset(inputs=x_test, labels=y_test)
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

from xgboost import XGBClassifier


model = XGBClassifier()
model.fit(x_train, y_train)


#------------------------------------------------------------------------------#
#------------------------------- experiment -----------------------------------#
#------------------------------------------------------------------------------#

from pnpxai import AutoExperiment


input_extractor = lambda batch: batch[0]
label_extractor = lambda batch: batch[1]
target_extractor = lambda outputs: outputs.argmax(-1)

expr = AutoExperiment(
    model=model,
    data=loader,
    modality='tabular',
    question='why',
    evaluator_enabled=False,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    background_data=x_train,
)

print('''
#------------------------------------------------------------------------------#
#---------------------------- recommender output ------------------------------#
#------------------------------------------------------------------------------#
''')
expr.recommended.print_tabular()

expr.run(data_ids=[0, 1], explainer_ids=range(len(expr.all_explainers)), metrics_ids=range(len(expr.all_metrics)))
expr.records[0]

from pprint import pprint

print('''
#------------------------------------------------------------------------------#
#--------------------------- experiment.records[0] ----------------------------#
#------------------------------------------------------------------------------#
''')
pprint(expr.records[0], sort_dicts=False)
# self._rearrange_explanations(),
# self._rearrange_evaluations(),

# expr.records[0]
import sys
sys.path.append('.')

import xgboost

from sklearn.datasets import load_iris

from xai_pnp import Project
from xai_pnp.explainers.shap import KernelShap
import shap
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris['data'], iris['target']
model = xgboost.XGBClassifier().fit(X, y)


project = Project('test_project')
exp = project.make_experiment(model, KernelShap(X))

input1 = X[10], y[10] # Label 0
input2 = X[60], y[60] # Label 1
input3 = X[100], y[100] # Label 2
exp.run(input1)
exp.run(input2)
exp.run(input3)

print(project.experiments)

# Return value is matplotlib figure.
# Transforming matplotlib figure to image and then 
# rendering image in plotly harms the image quaility

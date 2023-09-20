from typing import Any, List
# from shap import KernelExplainer as KE
import shap

from xai_pnp.core._types import Model, DataSource
from xai_pnp.explainers._explainer import Explainer
import numpy as np
import plotly.graph_objects as go


class Shap(Explainer):
    def __init__(self):
        super().__init__()
    
    def run(self, model: Model, data: DataSource, *args: Any, **kwargs: Any) -> List[np.array]:
        pass
    



class KernelShap(Explainer):
    def __init__(self, X):
        super().__init__()
        self.X = X
    
    def run(self, model: Model, data: DataSource, *args: Any, **kwargs: Any) -> List[np.array]:
        attributions = []
        explainer = shap.KernelExplainer(model.predict, self.X)
        shap_values = explainer.shap_values(data[0].reshape(1, -1))

        pred = model.predict(data[0].reshape(1, -1))[0]
        attributions.append(shap_values[0])
        fig = go.Figure(go.Waterfall(
            name = f"Prediction : {pred}",
            orientation = "v",
            measure = ["relative"for i in range(4)],
            x = [f"Feature {i}" for i in range(4)],
            textposition = "outside",
            y = shap_values[0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title = f"KernelSHAP (Label {data[1]})",
            showlegend = True
        )
        fig.show()

        return attributions

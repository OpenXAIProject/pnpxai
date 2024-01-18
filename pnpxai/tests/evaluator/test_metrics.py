from abc import abstractmethod
import pytest
import torch
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator import Complexity, MuFidelity, Sensitivity
from pnpxai.tests.helpers import ToyCNN, ToyExplainer, get_test_input_image

class _TestMetric:
    @pytest.fixture
    def model(self):
        return ToyCNN()
    
    @pytest.fixture
    def valid_input(self):
        return get_test_input_image(batch=True)
    
    @pytest.fixture
    def explainer(self, model):
        return ToyExplainer(model)
    
    @pytest.fixture
    def explainer_w_args(self, explainer):
        return ExplainerWArgs(explainer=explainer)
    
    @pytest.fixture
    def attributions(self, explainer, valid_input):
        return explainer.attribute(valid_input, 0)

    @abstractmethod
    @pytest.fixture
    def metric(self):
        return NotImplementedError
    
    def test_call_metric(self, metric, model, explainer_w_args, valid_input, attributions):
        metric(model, explainer_w_args, valid_input, torch.tensor([0]), attributions)
    
    def test_call_metric_without_optionals(self, metric, model, explainer_w_args, valid_input):
        metric(model, explainer_w_args, valid_input)


class TestComplexity(_TestMetric):
    @pytest.fixture
    def metric(self):
        return Complexity()

class TestMuFidelity(_TestMetric):
    @pytest.fixture
    def metric(self):
        return MuFidelity()

class TestSensitivity(_TestMetric):
    @pytest.fixture
    def metric(self):
        return Sensitivity()
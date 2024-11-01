from abc import abstractmethod
import pytest
import torch
from pnpxai.explainers import Explainer
from pnpxai.evaluator.metrics import Complexity, MuFidelity, Sensitivity
from tests.helpers import ToyCNN, ToyExplainer, get_test_input_image


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
    def attributions(self, explainer, valid_input):
        return explainer.attribute(valid_input, 0)

    @pytest.fixture()
    def valid_target(self):
        return torch.tensor([0])

    @abstractmethod
    @pytest.fixture
    def metric_type(self):
        return NotImplementedError

    def test_call_metric(
        self, metric_type, model, explainer, valid_input, valid_target, attributions
    ):
        metric = metric_type(model, explainer)
        metric.evaluate(valid_input, valid_target, attributions)

    def test_call_metric_without_optionals(
        self, metric_type, model, explainer, valid_input, valid_target
    ):
        metric = metric_type(model, explainer)
        metric.evaluate(valid_input, valid_target)


class TestComplexity(_TestMetric):
    @pytest.fixture
    def metric_type(self):
        return Complexity


class TestMuFidelity(_TestMetric):
    @pytest.fixture
    def metric_type(self):
        return MuFidelity


class TestSensitivity(_TestMetric):
    @pytest.fixture
    def metric_type(self):
        return Sensitivity

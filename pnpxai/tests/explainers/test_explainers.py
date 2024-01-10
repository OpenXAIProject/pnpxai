from abc import abstractmethod
import pytest
import torch

from pnpxai.explainers import (
    GradCam, GuidedGradCam, Lime, KernelShap,
    IntegratedGradients, LRP, RAP, ExplainerWArgs
)
from pnpxai.tests.helpers import ToyCNN, get_test_input_image

class _TestExplainer:
    @pytest.fixture
    def model(self):
        return ToyCNN()
    
    @pytest.fixture
    def valid_input(self):
        return get_test_input_image(batch=True)

    @abstractmethod
    @pytest.fixture
    def explainer_type(self):
        return NotImplementedError
    
    @pytest.fixture
    def explainer(self, explainer_type, model):
        return explainer_type(model)

    @pytest.fixture
    def explainer_w_args(self, explainer):
        return ExplainerWArgs(explainer)
    
    def test_attribute(self, explainer, valid_input):
        explainer.attribute(valid_input, 0)

    def test_attribute_with_invalid_inputs(self, explainer):
        # all explainers raise RuntimeError when forwarding input through model
        # if input is not valid.
        sizes = [
            (2, 2), # lesser dim case
            (1, 2, 2), # wrong channel case
            (1, 1, 2, 2), # larger dim case
        ]
        for size in sizes:
            invalid_input = get_test_input_image(size=size)
            with pytest.raises(RuntimeError):
                explainer.model(invalid_input)


class TestGradCam(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return GradCam
    
    def test_find_target_layer(self, model, explainer):
        assert explainer.source.layer is model.target_layer
    
    def test_find_target_layer_raises_without_pool(self, explainer_type):
        model_without_pool = ToyCNN(with_pool=False)
        with pytest.raises(AssertionError):
            explainer_type(model_without_pool)
    

class TestGuidedGradCam(TestGradCam):
    pass


class TestLime(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return Lime


class TestKernelShap(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return KernelShap
    
    @pytest.fixture
    def valid_feature_mask(self):
        # valid <=> n superpixels > 1
        return torch.eye(2).long().unsqueeze(0)

    def test_attribute(self, explainer, valid_input, valid_feature_mask):
        explainer.attribute(valid_input, 0, feature_mask=valid_feature_mask)
    
    def test_attribute_raises_with_invalid_feature_mask(self, explainer, valid_input):
        # Default feature mask has only one superpixel for the test valid_input.
        # Finally, ValueError occurs from captum.
        with pytest.raises(AssertionError):
            explainer.attribute(valid_input, 0)


class TestIntegratedGradients(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return IntegratedGradients


class TestLRP(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return LRP


class TestRAP(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return RAP

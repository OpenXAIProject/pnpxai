from abc import abstractmethod
import pytest
import torch

from pnpxai.explainers import (
    GradCam, GuidedGradCam, Lime, KernelShap,
    IntegratedGradients, LRP, RAP, ExplainerWArgs
)
from helpers import _TestModelCNN, get_test_input_image

class _TestExplainer:
    @pytest.fixture
    def model(self):
        return _TestModelCNN()
    
    @pytest.fixture
    def input(self):
        return get_test_input_image()

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
    
    def test_attribute(self, explainer, input):
        explainer.attribute(input, 0)


class TestGradCam(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return GradCam
    
    def test_find_target_layer(self, model, explainer):
        assert explainer.source.layer is model.target_layer
    
    def test_find_target_layer_raises_without_pool(self, explainer_type):
        model_without_pool = _TestModelCNN(with_pool=False)
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

    def test_attribute(self, explainer, input, valid_feature_mask):
        explainer.attribute(input, 0, feature_mask=valid_feature_mask)
    
    def test_attribute_raises_with_invalid_feature_mask(self, explainer, input):
        # Default feature mask has only one superpixel for the test input.
        # Finally, ValueError occurs from captum without AssertionError.
        with pytest.raises(AssertionError):
            explainer.attribute(input, 0)


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

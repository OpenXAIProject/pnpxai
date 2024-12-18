from abc import abstractmethod
import pytest
import torch

from pnpxai.explainers import (
    GradCam,
    GuidedGradCam,
    Lime,
    KernelShap,
    IntegratedGradients,
    LRPEpsilonAlpha2Beta1,
    LRPEpsilonGammaBox,
    LRPEpsilonPlus,
    LRPUniformEpsilon,
    RAP,
    Gfgp
)
from tests.helpers import ToyCNN, get_test_input_image


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

    def test_attribute(self, explainer, valid_input):
        explainer.attribute(valid_input, 0)

    def test_attribute_with_invalid_inputs(self, explainer):
        # all explainers raise RuntimeError when forwarding input through model
        # if input is not valid.
        sizes = [
            (2, 2),  # lesser dim case
            (1, 2, 2),  # wrong channel case
            (1, 1, 2, 2),  # larger dim case
        ]
        for size in sizes:
            invalid_input = get_test_input_image(size=size)
            with pytest.raises(RuntimeError):
                explainer.model(invalid_input)


class TestGradCam(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return GradCam


class TestGuidedGradCam(TestGradCam):
    @pytest.fixture
    def explainer_type(self):
        return GuidedGradCam


class TestLime(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return Lime


class TestKernelShap(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return KernelShap

    def test_attribute(self, explainer_type, model, valid_input):
        explainer = explainer_type(model, feature_mask_fn=[None])
        explainer.attribute(valid_input, 0)


class TestIntegratedGradients(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return IntegratedGradients


class TestLRP(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return LRPUniformEpsilon


class TestRAP(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return RAP

class TestGFGP():
    def test_model_loading(self):
        model = ToyCNN()
        transforms = lambda x: x
        explainer = Gfgp(model, transforms)
        assert explainer.diffusion_model is not None
        assert explainer.diffusion is not None
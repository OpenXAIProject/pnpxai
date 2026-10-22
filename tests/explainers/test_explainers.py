from abc import abstractmethod
import copy
import pytest
import torch
import torchvision

from pnpxai.core.modality import ImageModality, TextModality, TimeSeriesModality
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
    AE,
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


class TestAE(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return AE

    @pytest.fixture
    def resnet(self):
        torch.manual_seed(0)
        return torchvision.models.resnet18().eval()

    def test_attribute_matches_input(self, explainer_type, resnet):
        inputs = torch.randn(2, 3, 64, 64)
        attrs = explainer_type(resnet).attribute(inputs, torch.tensor([1, 2]))
        assert attrs.shape == inputs.shape
        assert attrs.dtype == inputs.dtype
        assert attrs.device == inputs.device
        assert torch.isfinite(attrs).all()

    def test_attribute_is_independent_of_batch(self, explainer_type, resnet):
        # Experiment caches explanations per sample
        inputs = torch.randn(3, 3, 64, 64)
        explainer = explainer_type(resnet)
        batched = explainer.attribute(inputs)
        single = torch.cat([explainer.attribute(x[None]) for x in inputs])
        torch.testing.assert_close(batched, single)

    def test_attribute_after_deepcopy(self, explainer_type, resnet):
        # Experiment.optimize returns optuna's deep copy of the best explainer
        inputs = torch.randn(1, 3, 64, 64)
        explainer = explainer_type(resnet, ratio=1.5)
        expected = explainer.attribute(inputs, 0)
        torch.testing.assert_close(copy.deepcopy(explainer).attribute(inputs, 0), expected)

    def test_tunable_ratio(self, explainer_type, resnet):
        inputs = torch.randn(1, 3, 64, 64)
        explainer = explainer_type(resnet)
        assert set(explainer.get_tunables()) == {"ratio"}
        tuned = explainer.set_kwargs(ratio=1.0)
        assert not torch.allclose(tuned.attribute(inputs, 0), explainer.attribute(inputs, 0))

    def test_image_modality_only(self, explainer_type):
        assert explainer_type in ImageModality.EXPLAINERS
        assert explainer_type not in TextModality.EXPLAINERS
        assert explainer_type not in TimeSeriesModality.EXPLAINERS


class TestGFGP():
    def test_model_loading(self):
        model = ToyCNN()
        transforms = lambda x: x
        explainer = Gfgp(model, transforms)
        assert explainer.diffusion_model is not None
        assert explainer.diffusion is not None
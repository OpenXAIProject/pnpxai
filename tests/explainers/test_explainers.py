from abc import abstractmethod
import pytest
import torch
import torchvision
from torch import nn

from pnpxai.core.modality import ImageModality, TextModality, TimeSeriesModality
from pnpxai.core.recommender import XaiRecommender
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
    Gfgp,
    UCAG,
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


class _ConvAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        tokens = self.conv(x).flatten(2).transpose(1, 2)
        return self.fc(self.attention(tokens, tokens, tokens)[0].mean(1))


class TestUCAG(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return UCAG

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return torchvision.models.resnet18().eval()

    @pytest.fixture
    def valid_input(self):
        return torch.randn(1, 3, 64, 64)

    def test_attribute_matches_input(self, explainer):
        inputs = torch.randn(2, 3, 64, 64)
        attrs = explainer.attribute(inputs, torch.tensor([1, 2]))
        assert attrs.shape == (2, 1, 64, 64)
        assert attrs.dtype == inputs.dtype
        assert attrs.isfinite().all() and attrs.min() >= 0 and attrs.max() <= 1

    def test_attribute_is_independent_of_batch(self, explainer):
        inputs = torch.randn(2, 3, 64, 64)
        batched = explainer.attribute(inputs, 3)
        single = torch.cat([explainer.attribute(x[None], 3) for x in inputs])
        torch.testing.assert_close(batched, single)

    def test_tunables(self, explainer, valid_input):
        assert set(explainer.get_tunables()) == {"alpha", "num_units"}
        tuned = explainer.set_kwargs(num_units=2)
        assert not torch.allclose(tuned.attribute(valid_input, 0), explainer.attribute(valid_input, 0))

    def test_recommended_like_cam_explainers(self, explainer_type, model):
        assert explainer_type in ImageModality.EXPLAINERS
        assert explainer_type not in TextModality.EXPLAINERS
        assert explainer_type not in TimeSeriesModality.EXPLAINERS
        recommender = XaiRecommender()
        for network, expected in ((model, True), (_ConvAttention(), False)):
            recommended = recommender.recommend(ImageModality(), network).explainers
            assert (GradCam in recommended) is expected
            assert (explainer_type in recommended) is expected


class TestGFGP():
    def test_model_loading(self):
        model = ToyCNN()
        transforms = lambda x: x
        explainer = Gfgp(model, transforms)
        assert explainer.diffusion_model is not None
        assert explainer.diffusion is not None
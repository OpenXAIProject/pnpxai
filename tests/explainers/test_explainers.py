from abc import abstractmethod
import copy
import pytest
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

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
    Gfgp,
    CoIBA,
    AVAILABLE_EXPLAINERS,
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


class TestCoIBA(_TestExplainer):
    @pytest.fixture
    def explainer_type(self):
        return CoIBA

    @staticmethod
    def calibration_loader(shape, n=8):
        return DataLoader(TensorDataset(torch.randn(n, *shape), torch.zeros(n).long()), batch_size=4)

    @pytest.fixture
    def explainer(self, explainer_type, model, valid_input):
        explainer = explainer_type(model, layer="conv")
        return explainer.estimate(self.calibration_loader(valid_input.shape[1:]), n_samples=8)

    @pytest.fixture
    def resnet_explainer(self, explainer_type):
        torch.manual_seed(0)
        explainer = explainer_type(torchvision.models.resnet18().eval(), layer=["layer3", "layer4"])
        return explainer.estimate(self.calibration_loader((3, 64, 64)), n_samples=8)

    def test_attribute_requires_preparation(self, explainer_type, model, valid_input):
        with pytest.raises(RuntimeError):
            explainer_type(model, layer="conv").attribute(valid_input, 0)

    def test_attribute_matches_input(self, resnet_explainer):
        inputs = torch.randn(2, 3, 64, 64)
        attrs = resnet_explainer.attribute(inputs, torch.tensor([1, 2]))
        assert attrs.shape == (2, 1, 64, 64)
        assert attrs.dtype == inputs.dtype
        assert attrs.isfinite().all() and attrs.min() >= 0 and attrs.max() <= 1

    def test_attribute_is_independent_of_batch(self, resnet_explainer):
        inputs = torch.randn(2, 3, 64, 64)
        batched = resnet_explainer.attribute(inputs, 3)
        single = torch.cat([resnet_explainer.attribute(x[None], 3) for x in inputs])
        torch.testing.assert_close(batched, single)

    def test_attribute_after_deepcopy(self, resnet_explainer):
        # Experiment.optimize returns optuna's deep copy of the best explainer
        inputs = torch.randn(1, 3, 64, 64)
        expected = resnet_explainer.attribute(inputs, 0)
        torch.testing.assert_close(copy.deepcopy(resnet_explainer).attribute(inputs, 0), expected)

    def test_tunable_beta_keeps_statistics(self, resnet_explainer):
        inputs = torch.randn(1, 3, 64, 64)
        assert set(resnet_explainer.get_tunables()) == {"beta"}
        tuned = resnet_explainer.set_kwargs(beta=0.01)
        assert not torch.allclose(tuned.attribute(inputs, 0), resnet_explainer.attribute(inputs, 0))
        with pytest.raises(RuntimeError):
            resnet_explainer.set_kwargs(layer="layer4").attribute(inputs, 0)

    def test_statistics_round_trip(self, resnet_explainer, explainer_type, tmp_path):
        inputs = torch.randn(1, 3, 64, 64)
        path = tmp_path / "coiba.pt"
        resnet_explainer.save_statistics(path, preprocessing_id="test")
        restored = explainer_type(resnet_explainer.model, layer=["layer3", "layer4"])
        restored.load_statistics(path, preprocessing_id="test")
        torch.testing.assert_close(restored.attribute(inputs, 0), resnet_explainer.attribute(inputs, 0))

    def test_not_recommended(self, explainer_type):
        # attribute() needs layers and calibration statistics first
        assert explainer_type not in AVAILABLE_EXPLAINERS
        for modality in (ImageModality, TextModality, TimeSeriesModality):
            assert explainer_type not in modality.EXPLAINERS


class TestGFGP():
    def test_model_loading(self):
        model = ToyCNN()
        transforms = lambda x: x
        explainer = Gfgp(model, transforms)
        assert explainer.diffusion_model is not None
        assert explainer.diffusion is not None
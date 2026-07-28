from abc import abstractmethod
from types import SimpleNamespace
import pytest
import torch
from torch import nn

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
    MAGIG,
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


class StubAutoencoder(nn.Module):
    """
    Smallest thing shaped like a `diffusers` autoencoder.

    Lets MAGIG be tested end to end -- path construction, latent gradients,
    accumulation -- without pulling a real VAE off the Hub.
    """

    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        self.enc = nn.Conv2d(in_channels, latent_channels, 1)
        self.dec = nn.Conv2d(latent_channels, in_channels, 1)

    @property
    def dtype(self):
        return self.enc.weight.dtype

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.enc(x)))

    def decode(self, z):
        return SimpleNamespace(sample=self.dec(z))


class TestMAGIG(_TestExplainer):
    @pytest.fixture
    def explainer(self, model):
        return MAGIG(model, vae=StubAutoencoder(), n_steps=4)

    def test_path_endpoints_are_anchored(self, explainer, valid_input):
        """The path must start at the baseline and end at the input exactly."""
        targets = torch.zeros(1, dtype=torch.long)
        path = explainer.generate_path(valid_input, targets)
        baseline = explainer.vae.normalize(torch.zeros_like(valid_input))
        assert path.shape[0] == explainer.n_steps
        assert torch.equal(path[0], baseline[0])
        assert torch.equal(path[-1], valid_input[0])

    def test_attribute_shape_matches_input(self, explainer, valid_input):
        attrs = explainer.attribute(valid_input, torch.zeros(1, dtype=torch.long))
        assert attrs.shape == valid_input.shape

    def test_attribute_is_deterministic(self, explainer, valid_input):
        targets = torch.zeros(1, dtype=torch.long)
        first = explainer.attribute(valid_input, targets)
        second = explainer.attribute(valid_input, targets)
        assert torch.equal(first, second)

    def test_batch_matches_per_sample(self, explainer):
        """Samples are explained independently, so a batch equals its parts."""
        batch = torch.cat([get_test_input_image() for _ in range(3)])
        targets = torch.tensor([0, 1, 0])
        batched = explainer.attribute(batch, targets)
        for i in range(batch.shape[0]):
            alone = explainer.attribute(batch[i][None], targets[i][None])
            assert torch.equal(batched[i][None], alone)

    def test_rejects_unknown_objective(self, model):
        with pytest.raises(ValueError):
            MAGIG(model, vae=StubAutoencoder(), exp_obj="nonsense")

    def test_no_determinism_warning_on_cpu(self, model):
        """The cuDNN determinism warning is CUDA-specific; CPU users shouldn't see it."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MAGIG(model, vae=StubAutoencoder(), n_steps=4)
        assert not [w for w in caught if "cudnn" in str(w.message).lower()]

    def test_baseline_fn_accepts_pnpxai_forms(self, model, valid_input):
        """None means the paper's black image; strings and functions go through
        the base class's baseline machinery like any other explainer."""
        from pnpxai.explainers.utils.baselines import ZeroBaselineFunction
        from pnpxai.explainers.magig import IMAGENET_MEAN, IMAGENET_STD

        black = MAGIG(model, vae=StubAutoencoder(), n_steps=4)._baselines_for(valid_input)
        expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
        assert torch.allclose(black[0, :, 0, 0], expected)

        for baseline_fn in ("zeros", ZeroBaselineFunction()):
            explainer = MAGIG(model, vae=StubAutoencoder(), n_steps=4,
                              baseline_fn=baseline_fn)
            assert torch.equal(explainer._baselines_for(valid_input),
                               torch.zeros_like(valid_input))
            assert explainer.attribute(
                valid_input, torch.zeros(1, dtype=torch.long)
            ).shape == valid_input.shape

    def test_tunables_are_suggestable(self, explainer):
        """Every tunable must be a type the optimizer's suggestor can sample."""
        import optuna
        from pnpxai.core.modality.modality import ImageModality
        from pnpxai.evaluator.optimizer.suggestor import suggest

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        modality = ImageModality()

        def objective(trial):
            tuned = suggest(trial, explainer, modality)
            assert isinstance(tuned.n_steps, int)
            assert isinstance(tuned.use_slerp, bool)
            return float(tuned.n_steps)

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3
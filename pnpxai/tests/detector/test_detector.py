import pytest
import random
from torch import nn

from pnpxai.detector import ModelArchitectureDetector

TEST_LAYERS = [
    nn.Conv2d(1,1,1),
    nn.BatchNorm2d(1,1),
    nn.ReLU(),
    nn.Linear(1,1),
]

class TestModelArchitectureDetector:
    @property
    def n_seq(self):
        return len(TEST_LAYERS)+1

    @property
    def random_sequence(self):
        return [random.choice(TEST_LAYERS) for _ in range(self.n_seq)]    

    @pytest.fixture
    def detector(self):
        return ModelArchitectureDetector()    
    
    @pytest.fixture
    def model(self):
        return nn.Sequential(*self.random_sequence)
        
    def test_extract_modules(self, detector, model):
        init_mode = model.training
        detector.extract_modules(model)
        assert init_mode == model.training
        assert len(detector.modules) == self.n_seq


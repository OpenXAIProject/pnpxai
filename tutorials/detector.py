from _operator import add
import torchvision
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.recommender import XaiRecommender

model = torchvision.models.get_model("vit_b_16").eval()

detector = ModelArchitectureDetector()
detector_output = detector(model, sample=None)

recommender = XaiRecommender()
applicables = recommender.filter_methods("why", "image", detector_output.architecture)
print(applicables)

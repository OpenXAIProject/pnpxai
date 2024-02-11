from _operator import add
import torchvision
import timm
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.recommender import XaiRecommender

model = timm.create_model("vit_small_patch8_224")
detector = ModelArchitectureDetector()
recommender = XaiRecommender()
applicables = recommender.filter_methods("why", "image", detector(model))
print(applicables)

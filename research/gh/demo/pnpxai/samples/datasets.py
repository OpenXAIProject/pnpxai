import os
import requests
import subprocess

from torch.utils.data import Dataset
from torchvision.io import read_image


def git_clone_imagenet_sample_images():
    subprocess.check_call(["git", "clone", "https://github.com/EliSchwartz/imagenet-sample-images.git"])


def get_imagenet_label_map():
    res = requests.get('https://gist.githubusercontent.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372/raw/d25516d26be4a8d3e0aeebe9275631754b8e2c73/imagenet_label_to_wordnet_synset.txt')
    label_map = eval(res.text)
    char_to_num = dict()
    for numid, v in label_map.items():
        charid = v.pop("id")
        charid = ''.join(reversed(charid.split('-')))
        char_to_num[charid] = numid
    return char_to_num


class ImageNetSample(Dataset):
    IMG_DIR = "imagenet-sample-images"
    LABEL_MAP = get_imagenet_label_map()
    
    def __init__(self, transform=None):
        if not os.path.exists(self.IMG_DIR):
            git_clone_imagenet_sample_images()
        self.transform = transform
        self.filenames = [filename for filename in os.listdir(self.IMG_DIR) if filename.endswith(".JPEG")]

    def _get_image_path(self, filename):
        return os.path.join(self.IMG_DIR, filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.IMG_DIR, img_filename)
        x = read_image(img_path)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1) # gray to rgb
        if self.transform:
            x = self.transform(x)
        charid = img_path.split('/')[-1].split('_')[0]
        y = self.LABEL_MAP[charid]
        return x, y
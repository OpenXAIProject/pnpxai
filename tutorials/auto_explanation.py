import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torchvision.models.resnet import resnet18, ResNet18_Weights

from open_xai.utils import set_seed
from open_xai import Project, AutoExplanationInput

import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#-----------------------------------------------------------------------------#
#----------------------------------- model -----------------------------------#
#-----------------------------------------------------------------------------#

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)


#-----------------------------------------------------------------------------#
#------------------------------------ data -----------------------------------#
#-----------------------------------------------------------------------------#

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'samples/')
        self.label_dir = os.path.join(self.root_dir, 'imagenet_class_index.json')
        
        with open(self.label_dir) as json_data:
            self.idx_to_labels = json.load(json_data)
        
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = idx
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def idx_to_label(self, idx):
        return self.idx_to_labels[str(idx)][1]

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

imagenet_data = ImageNetDataset(root_dir='./data/ImageNet/', transform=data_transforms)
loaders = {
    'test': torch.utils.data.DataLoader(
        imagenet_data, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    ),
}


#-----------------------------------------------------------------------------#
#---------------------------------- explain ----------------------------------#
#-----------------------------------------------------------------------------#

user_input = AutoExplanationInput(
    model=model,
    data=loaders['test'],
    question='why',
    task='image',
    input_extractor=lambda x: x[0].to(device),
    label_extractor=lambda x: x[1].to(device),
)

# pdb.set_trace()

pnp_xai_project = Project('test_project')
exp = pnp_xai_project.auto_explain(user_input)
print(exp.runs[0].explanation[0])
print(exp.runs[0].evaluation)
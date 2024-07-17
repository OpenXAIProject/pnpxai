import os
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


class LiverTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        # self.samples = np.load(os.path.join(data_dir, 'samples.npz'))['samples']
        # self.labels = np.load(os.path.join(data_dir, 'labels.npz'))['labels']
        self.samples = np.load(os.path.join(data_dir, 'samples_subset.npz'))['samples'].astype(np.float32)
        self.labels = np.load(os.path.join(data_dir, 'labels_subset.npz'))['labels']
        self.labels = torch.from_numpy(self.labels).to(torch.int64)

        # self.labels = F.one_hot(self.labels)
        # self.masks = np.load(os.path.join(data_dir, 'masks.npz'))['masks']
        self.masks = np.load(os.path.join(data_dir, 'masks_subset.npz'))['masks']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.from_numpy(sample).float()

        return sample, label, mask

    def idx_to_label(self, idx):
        # label = self.labels.argmax(dim=1)[idx].item()
        # label = self.labels[idx].item()
        label = self.labels[idx.item()]
        print(f'label: {label} / idx: {idx}')
        if label == 0:
            return 'Normal'
        elif label == 1:
            return 'Tumor'

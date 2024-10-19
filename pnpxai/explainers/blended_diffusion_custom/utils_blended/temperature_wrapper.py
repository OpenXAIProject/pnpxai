import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from RATIO_utils.config import model_type_to_folder, model_name_to_folder
import os
import pickle

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def get_temperature_folder(dataset, type_, folder):
    return os.path.join(here, '../', model_name_to_folder[dataset.lower()],
                 model_type_to_folder[type_.lower()],
                 folder)


class TemperatureWrapper(nn.Module):
    def __init__(self, model, T=1.):  #, noise_sigma=0.2, noise_iterations=100)#, randomized_smoothing=False):
        super().__init__()

        # ToDo: do depending on the config
        # For CLIP comment it out
        #self.train(model.training)

        self.model = model
        print('Temperature is', T)
        self.T = T
        #self.noise_sigma = noise_sigma
        #self.randomized_smoothing = randomized_smoothing
        #self.noise_iterations = noise_iterations


    def forward(self, x):
        """
        if self.randomized_smoothing:
            print('randomized smoothing on')
            ret = 0
            for i in range(self.noise_iterations):
                print(i)
                x_noisy = x + self.noise_sigma * torch.randn_like(x)
                logits = self.model(x_noisy)
                ret += logits #/ self.T
            return ret / self.noise_iterations
        else:
        """
        #print('randomized smoothing off')
        #print('Temperature is', self.T)
        logits = self.model(x) # For CLIP uncomment it .type(torch.float32))
        if self.model.model.return_layers:
            return logits
        else:
            return logits / self.T

    @staticmethod
    def compute_temperature(model, loader, batch_size, device, type_, folder, dataset, img_size, project_folder, data_folder):
        if 'CLIP' in type_:
            pass
        else:
            model.eval()

        logits = []
        labels = []

        temperature_folder = get_temperature_folder(dataset, type_, folder)

        temperature_file = os.path.join(temperature_folder,
                                        'temperature.pickle')

        if not os.path.exists(temperature_folder):
            os.makedirs(temperature_folder)

        if os.path.isfile(temperature_file):
            print('Loading temperature file', temperature_file)
            with open(temperature_file, 'rb') as f:
                T = pickle.load(f)

            print('Temperature is', T)
            return T
        else:
            print('Saving temperature file', temperature_file)
            with torch.no_grad():
                for out in tqdm(loader(batch_size, img_size, project_folder=project_folder, data_folder=data_folder, model_name=type_)[dataset.lower()]):
                    if len(out) == 2:
                        data, target = out
                    elif len(out) == 3:
                        data, target, _ = out
                    data = data.to(device)
                    #print(data.shape, data.dtype, data)
                    logits.append(model(data).detach().cpu()) # For CLIP uncomment it .type(torch.float32))
                    labels.append(target)

            logits = torch.cat(logits, 0)
            labels = torch.cat(labels, 0)

            ca = []
            log_T = torch.linspace(-3., 3., 2000)
            log_T = torch.cat((log_T, torch.Tensor([0])))

            for t in log_T:
                #print('temperature', t)
                ca.append(TemperatureWrapper.get_ece_inner(logits / np.exp(t), labels)[0])
            ece, idx = torch.stack(ca, 0).min(0)

            T = float(np.exp(log_T[idx]))

            with open(temperature_file, 'wb+') as f:
                pickle.dump(T, f)

            print('Temperature is', T)
            return T


    @staticmethod
    def compute_ece(model, loader, device):
        model.eval()
        logits = []
        labels = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)

                logits.append(model(data).detach().cpu())
                labels.append(target)

        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)
        ece = TemperatureWrapper.get_ece_inner(logits, labels)[0]
        return ece

    @staticmethod
    def get_ece_inner(logits, labels, n_bins=20):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        #print(logits.shape, logits)
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


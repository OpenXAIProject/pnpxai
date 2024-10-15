import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pnpxai.core.modality import ImageModality
from pnpxai.explainers import IntegratedGradients
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.explainers.utils.baselines import ZeroBaselineFunction
from pnpxai.evaluator.metrics import AbPC
from pnpxai.evaluator.optimizer import optimize, Objective
from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


torch.set_num_threads(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, transform = get_torchvision_model('resnet18')
model.to(device)
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, shuffle=False, batch_size=1)

inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)
preds = outputs.argmax(-1)

explainer = IntegratedGradients(
    model=model,
    baseline_fn=ZeroBaselineFunction(),
)
postprocessor = PostProcessor.from_name(
    pooling_method='l2normsq',
    normalization_method='minmax',
    channel_dim=1,
)
metric = AbPC(
    model=model,
    baseline_fn=ZeroBaselineFunction(),
)
obj = Objective(
    explainer=explainer,
    postprocessor=postprocessor,
    metric=metric,
    modality=ImageModality(channel_dim=1),
)

study = optimize(
    objective=obj.set_data(inputs, preds),
    sampler='random',
    n_trials=30,
    direction='maximize',
    seed=42,
)

best_explainer = study.best_trial.user_attrs['explainer']
best_postprocessor = study.best_trial.user_attrs['postprocessor']

attrs = best_explainer.attribute(inputs, preds)
best = best_postprocessor(attrs)

worst_trial = next(iter(sorted(
    filter(lambda trial: trial.value is not None, study.trials),
    key=lambda trial: trial.value
)))
worst_explainer = worst_trial.user_attrs['explainer']
worst_postprocessor = worst_trial.user_attrs['postprocessor']

attrs = worst_explainer.attribute(inputs, preds)
worst = worst_postprocessor(attrs)

fig, axes = plt.subplots(1, 3)
img = denormalize_image(inputs[0].cpu().detach(), mean=transform.mean, std=transform.std)
axes[0].imshow(img)
axes[0].set_title(explainer.__class__.__name__)
axes[1].imshow(worst[0].cpu().detach().numpy(), cmap='coolwarm')
axes[1].set_title(f'Worst: {format(worst_trial.value, ".4f")}')
axes[2].imshow(best[0].cpu().detach().numpy(), cmap='coolwarm')
axes[2].set_title(f'Best: {format(study.best_trial.value, ".4f")}')

for ax in axes.flat:
    ax.axis('off')

fig.tight_layout()
fig.savefig('optimize_example.png')
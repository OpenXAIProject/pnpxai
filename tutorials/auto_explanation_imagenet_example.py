from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pnpxai import AutoExplanation

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


# setup
model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# create auto explanation
expr = AutoExplanation(
    model=model,
    data=loader,
    modality='image',
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    target_labels=False, # target prediction if False
)

# browse the recommended
expr.recommended.print_tabular() # recommendation
expr.recommended.explainers # -> List[Type[Explainer]]

# browse explainers and metrics
expr.manager.explainers # -> List[Explainer]
expr.manager.metrics # -> List[Metric]

expr.manager.get_explainer_by_id(7) # -> Explainer. In this case, LRPEpsilonGammaBox
expr.manager.get_postprocessor_by_id(0) # -> PostProcessor. In this case, PostProcessor(pooling_method='sumpos', normalization_method='minmax')
expr.manager.get_metric_by_id(0) # -> Metric. In this case, AbPC


# explain and evaluate
results = expr.run_batch(
    data_ids=[0],
    explainer_id=7,
    postprocessor_id=0,
    metric_id=0,
)

# optimize: returns optimal explainer id, optimal postprocessor id, (and study)
opt_explainer_id, opt_postprocessor_id, study = expr.optimize(
    data_id=0,
    explainer_id=7,
    metric_id=0, # metric to be the objective
    direction='maximize', # larger better
    sampler='tpe',
    n_trials=1, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
    return_study=True, # by default False. You can analyze the study by optuna
    seed=42, # seed for sampler: by default, None
)

# explain and evaluate with optimal explainer and postprocessor
opt_results = expr.run_batch(
    data_ids=[0],
    explainer_id=opt_explainer_id,
    postprocessor_id=opt_postprocessor_id,
    metric_id=0,
)


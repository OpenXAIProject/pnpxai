import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch.utils.data import DataLoader
import os
from pnpxai import AutoExplanationForTSClassification
from tsai.all import get_UCR_data, combine_split_data, Categorize, TSDatasets, TSDataLoaders, TSStandardize, InceptionTime, Learner, accuracy, TSTensor

# ------------------------------------------------------------------------------#
# -------------------------------- basic usage ---------------------------------#
# ------------------------------------------------------------------------------#

# setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)
device = torch.device('cpu')
cur_path = os.path.dirname(os.path.realpath(__file__))
batch_size = 64


def get_ts_dataset_loader(dataset: str, path: str, batch_size: int = 64):
    x_train, y_train, x_valid, y_valid = get_UCR_data(
        dataset, path, return_split=True
    )
    x, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    dsets = TSDatasets(
        x, y, tfms=[None, [Categorize()]], splits=splits, inplace=True
    )
    return TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=batch_size, batch_tfms=[TSStandardize()]
    )


def train_model(loader: TSDataLoaders, model: torch.nn.Module, model_path: str, epochs: int = 15, lr: float = 1e-3):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    learner = Learner(loader, model, metrics=accuracy)
    try:
        learner.load(model_path)
    except:
        learner.lr_find()
        plt.close()
        learner.fit_one_cycle(epochs, lr_max=lr)
        learner.save(model_path)
        torch.cuda.empty_cache()

    return learner.model


def tensor_mapper(x: TSTensor):
    return torch.from_numpy(x.cpu().numpy())


dsid = "TwoLeadECG"
loader = get_ts_dataset_loader(dsid, cur_path, batch_size)
model = InceptionTime(loader.vars, loader.c)
model = train_model(
    loader, model, f"{cur_path}/data/models/inceptiontime/{dsid}.pth", epochs=40)

test_data = DataLoader(
    loader.valid.dataset,
    batch_size=batch_size,
    shuffle=False
)

# create auto explanation
expr = AutoExplanationForTSClassification(
    model=model.to(device),
    data=test_data,
    input_extractor=lambda batch: tensor_mapper(batch[0]).to(device),
    label_extractor=lambda batch: tensor_mapper(batch[-1]).to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    target_labels=False,  # target prediction if False
)


# browse the recommended
expr.recommended.print_tabular()  # recommendation
expr.recommended.explainers  # -> List[Type[Explainer]]

# browse explainers and metrics
expr.manager.explainers  # -> List[Explainer]
expr.manager.metrics  # -> List[Metric]

# -> Explainer. In this case, LRPEpsilonGammaBox
expr.manager.get_explainer_by_id(0)
# -> PostProcessor. In this case, PostProcessor(pooling_method='sumpos', normalization_method='minmax')
expr.manager.get_postprocessor_by_id(0)
expr.manager.get_metric_by_id(3)  # -> Metric. In this case, AbPC

# user inputs
explainer_id = 4  # explainer_id to be optimized: KernelShap
metric_id = 1  # metric_id to be used as objective: AbPC
post_processor_id = 0
data_id = 0

explainer = expr.manager.get_explainer_by_id(explainer_id)
metric = expr.manager.get_metric_by_id(metric_id)
print("Explainer: ", explainer.__class__.__name__)
print("Metric: ", metric.__class__.__name__)

# explain and evaluate
results = expr.run_batch(
    data_ids=range(batch_size),
    explainer_id=explainer_id,
    postprocessor_id=post_processor_id,
    metric_id=metric_id,
)


# ------------------------------------------------------------------------------#
# ------------------------------- optimization ---------------------------------#
# ------------------------------------------------------------------------------#


# optimize: returns optimal explainer id, optimal postprocessor id, (and study)
optimized = expr.optimize(
    data_ids=[data_id],
    explainer_id=explainer_id,
    metric_id=metric_id,
    direction='maximize',  # larger better
    sampler='tpe',  # Literal['tpe','random']
    n_trials=50,
    seed=42,  # seed for sampler: by default, None
)

print('Best/Explainer:', optimized.explainer) # get the optimized explainer
print('Best/PostProcessor:', optimized.postprocessor) # get the optimized postprocessor
print('Best/value:', optimized.study.best_trial.value) # get the optimized value

# Every trial in study has its explainer and postprocessor in user attr.
i = 25
print(f'{i}th Trial/Explainer', optimized.study.trials[i].user_attrs['explainer']) # get the explainer of i-th trial
print(f'{i}th Trial/PostProcessor', optimized.study.trials[i].user_attrs['postprocessor']) # get the postprocessor of i-th trial
print(f'{i}th Trial/value', optimized.study.trials[i].value)

# For example, you can use optuna's API to get the explainer and postprocessor of the worst trial
def get_worst_trial(study):
    valid_trials = [trial for trial in study.trials if trial.value is not None]
    return sorted(valid_trials, key=lambda trial: trial.value)[0]

worst_trial = get_worst_trial(optimized.study)
print('Worst/Explainer:', worst_trial.user_attrs['explainer'])
print('Worst/PostProcessor', worst_trial.user_attrs['postprocessor'])
print('Worst/value', worst_trial.value)


# ------------------------------------------------------------------------------#
# ------------------------------- visualization --------------------------------#
# ------------------------------------------------------------------------------#

def threshold_plot(ax: Axes, predictions, confidences, color_mapper=None):
    # Define colors based on confidence RGB
    if color_mapper is None:
        def color_mapper(c):
            return (1, 1 - c, 1 - c) if c > 0 else (1 + c, 1 + c, 1)

    colors = [color_mapper(c) for c in confidences]

    # Plotting
    ax.plot(predictions, color='black', label='Lookback')
    min_pred = min(predictions)
    max_pred = max(predictions)
    for i in range(len(predictions) - 1):
        ax.fill_betweenx([min_pred, max_pred], i, i + 1, color=colors[i])


# plots
uniq_labels = {}
for idx, label in enumerate(expr.get_labels_flattened()):
    uniq_labels[label] = idx
    if len(uniq_labels) == loader.c:
        break

uniq_labels = dict(sorted(uniq_labels.items(), key=lambda x: x[0]))
fig, axes = plt.subplots(len(uniq_labels), 2, figsize=(4, 4))

# inputs
data_ids = list(uniq_labels.values())
data, _ = expr.manager.get_data(data_ids)
inputs, labels = [], []
for datum, label in data:
    inputs.append(datum)
    labels.append(label)

inputs = torch.concatenate(inputs).to(device)
labels = torch.concatenate(labels).to(device)

for idx, (data_id, label) in enumerate(uniq_labels.items()):
    plot_inputs = inputs[idx, 0, :].tolist()
    axes[idx, 0].plot(plot_inputs)
    axes[idx, 0].set_ylabel(f'Label: {label}')
    attrs = expr.manager.get_explanation_by_id(data_id, explainer_id)
    attrs = attrs / attrs.abs().max()
    threshold_plot(axes[idx, 1], plot_inputs, attrs[0, :].tolist())

plt.tight_layout()
plt.savefig(
    os.path.join(
        cur_path,
        f'opt_{explainer.__class__.__name__}_by_{metric.__class__.__name__}_all.png'
    ))


# plots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

opt_explainer_id = expr.manager.add_explainer(optimized.explainer)
opt_attrs = expr.explain_batch(
    data_ids=[data_id],
    explainer_id=opt_explainer_id,
)

# inputs
inputs, _ = expr.manager.batch_data_by_ids(data_ids=[data_id])
inputs = inputs.to(device)
targets = expr.manager.batch_outputs_by_ids(data_ids=[data_id])\
    .argmax(-1).to(device)

axes[0].plot(inputs[0, 0, :].tolist())

trials = [trial for trial in optimized.study.trials if trial.value is not None]
trials = sorted(trials, key=lambda trial: trial.value)
trials = {
    'worst': trials[0],  # worst
    'med': trials[len(trials)//2],  # med
    'best': trials[-1],  # best
}

for loc, (title, trial) in enumerate(trials.items(), 1):
    explainer = trial.user_attrs['explainer']
    postprocessor = trial.user_attrs['postprocessor']
    attrs = explainer.attribute(inputs, targets)
    postprocessed = postprocessor(attrs)
    axes[loc].set_title(f'{title}:{"{:4f}".format(trial.value)}')
    plot_atts = postprocessed / postprocessed.abs().max()
    threshold_plot(axes[loc], inputs[0, 0, :].tolist(),
                   plot_atts[0, 0, :].tolist())

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

metric = expr.manager.get_metric_by_id(metric_id)
plt.savefig(
    os.path.join(
        cur_path,
        f'opt_{explainer.__class__.__name__}_by_{metric.__class__.__name__}.png'
    ))

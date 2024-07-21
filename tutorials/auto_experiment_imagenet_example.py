from torch.utils.data import DataLoader
from pnpxai.core.experiment import AutoExperiment
from helpers import get_imagenet_dataset, get_torchvision_model

# create auto experiment
model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
expr = AutoExperiment(
    model=model,
    data=loader,
    modality='image',
    question='why',
    evaluator_enabled=True,
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    target_labels=False, # target prediction if False
)

# recommender output
expr.recommended.print_tabular()

# run expr
expr.run(
    data_ids=[0, 42],
    explainer_ids=range(len(expr.all_explainers)),
    metrics_ids=range(2),
)

# browse and visualize a record
import matplotlib.pyplot as plt
from pnpxai.explainers.utils.postprocess import postprocess_attr
from helpers import denormalize_image


def plot_record(record, pooling_method='l2normsq', normalization_method='minmax', savepath='./test.png'):
    label = dataset.dataset.idx_to_label(record['label'].item())
    pred = dataset.dataset.idx_to_label(record['output'].argmax().item())
    topk = record['output'].softmax(-1).topk(3)

    nrows, ncols = 1, 1+len(record['explanations'])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes[0].imshow(denormalize_image(record['input'], mean=transform.mean, std=transform.std))
    for ax, data in zip(axes[1:], record['explanations']):
        ax.imshow(
            postprocess_attr(
                data['value'],
                channel_dim=0,
                pooling_method=pooling_method,
                normalization_method=normalization_method
            ).detach().numpy(),
            cmap='coolwarm'
        )
        ax.set_title(data['explainer_nm'])
    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks([])
    fig.savefig(savepath)

plot_record(expr.records[1], savepath='./auto_experiment_imagenet_example.png')


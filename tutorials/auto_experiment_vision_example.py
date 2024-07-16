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
    explainer_ids=range(3),
    metrics_ids=range(2),
)

# browse record of results
record = expr.records[0]

print("record['data_id]:", record['data_id'])
print("record['input'].shape:", record['input'].shape)
print("record['target']:", record['target'])
print("record['explanations'][0]", record['explanations'][0])

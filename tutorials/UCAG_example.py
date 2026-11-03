"""Runnable companion to tutorials/UCAG_example.md.

Run from the repository root after installing pnpxai:

    python tutorials/UCAG_example.py --device cpu --threads 8

ResNet50 (IMAGENET1K_V1) explains four ImageNet sample images from
tutorials/data/ImageNet for their ground-truth classes; four other sample
images are used for a short hyperparameter search. Figures and results.json
are written to tutorials/data/ucag/. Python snippets in the Markdown are
copied from the DOC blocks below.
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import captum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from pnpxai import AutoExplanationForImageClassification
from pnpxai.evaluator.metrics import Complexity, MuFidelity
from pnpxai.explainers import UCAG
from pnpxai.explainers.utils import find_cam_target_layer
from pnpxai.explainers.utils.postprocess import Identity, SumPos
from pnpxai.utils import set_seed

from helpers import denormalize_image, get_imagenet_dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FINAL_IDS = [0, 1, 2, 3]  # one sample image per class, so labels are 0-3
TUNING_IDS = [100, 300, 500, 700]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "cpu"


def display(maps):
    """Per-image min-max for display; constant maps stay zero."""
    flat = maps.flatten(1)
    low, high = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
    return ((flat - low) / (high - low).clamp_min(1e-12)).view_as(maps)


def save_figure(images, maps, titles, path):
    count = len(images)
    fig, axes = plt.subplots(2, count, figsize=(2.0 * count, 4.4), squeeze=False)
    for i in range(count):
        image = denormalize_image(images[i].cpu(), IMAGENET_MEAN, IMAGENET_STD).clip(0, 1)
        axes[0, i].imshow(image)
        axes[0, i].set_title(titles[i], fontsize=9)
        axes[1, i].imshow(image)
        axes[1, i].imshow(maps[i].cpu(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", path, f"({path.stat().st_size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: PyTorch's choice)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--perturbations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    # get_imagenet_dataset() changes the working directory, so resolve first.
    output_dir = (args.output_dir or Path(__file__).resolve().parent / "data" / "ucag").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)

    seed, perturbations = args.seed, args.perturbations
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    # DOC: model_and_data
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = get_imagenet_dataset(transform, indices=FINAL_IDS + TUNING_IDS)
    loader = DataLoader(dataset, batch_size=4)
    final_ids, tuning_ids = [0, 1, 2, 3], [4, 5, 6, 7]  # positions in `dataset`
    # END DOC: model_and_data

    # DOC: auto_explanation
    experiment = AutoExplanationForImageClassification(
        model, loader, input_extractor=lambda batch: batch[0].to(device),
        label_extractor=lambda batch: batch[1].to(device),
        target_extractor=lambda outputs: outputs.argmax(-1).to(device),
        target_labels=True)
    assert UCAG in experiment.recommended.explainers
    # END DOC: auto_explanation
    default_id = next(i for i, explainer in enumerate(experiment.manager.explainers)
                      if isinstance(explainer, UCAG))

    # DOC: explainer
    uid = experiment.manager.add_explainer(UCAG(
        model=model, layer="layer4", crop_size=124, num_units=6,
        alpha=2.6, limit_batch_size=36, score=True,
    ))
    # END DOC: explainer

    # DOC: postprocessor
    pid = next(i for i, pp in enumerate(experiment.manager.postprocessors)
               if isinstance(pp.pooling_fn, SumPos) and isinstance(pp.normalization_fn, Identity))
    # END DOC: postprocessor

    # DOC: metrics
    metric_ids = [experiment.manager.add_metric(metric) for metric in (
        MuFidelity(model, n_perturb=perturbations), Complexity(model, n_bins=10))]
    # END DOC: metrics

    # DOC: run_batch
    scores = {}
    for metric_id in metric_ids:
        set_seed(seed)
        result = experiment.run_batch(explainer_id=uid, postprocessor_id=pid,
                                      metric_id=metric_id, data_ids=final_ids)
        scores[type(result["metric"]).__name__] = result["evaluation"].tolist()
    # END DOC: run_batch

    maps, pooled = result["explanation"], result["postprocessed"]
    inputs, targets = result["inputs"], result["targets"]
    assert maps.shape == (len(final_ids), 1, *inputs.shape[-2:]) and maps.dtype == inputs.dtype
    assert maps.isfinite().all() and maps.min() >= 0 and maps.max() <= 1
    assert pooled.shape == (len(final_ids), *inputs.shape[-2:])
    predictions = result["outputs"].argmax(-1).tolist()
    label = dataset.dataset.idx_to_label
    target_names = [label(t) for t in targets.tolist()]
    prediction_names = [label(p) for p in predictions]
    print("[3] targets", targets.tolist(), target_names, "| predictions", predictions, prediction_names)
    print("[4] maps", tuple(maps.shape), maps.dtype, "range", maps.min().item(), maps.max().item(),
          "| postprocessed", tuple(pooled.shape))
    titles = [f"{t}\npred: {p}" for t, p in zip(target_names, prediction_names)]
    save_figure(inputs, display(pooled), titles, output_dir / "ucag.png")

    # The recommended instance (layer=None, crop batch 16) discovers its layer.
    discovered = next(name for name, module in model.named_modules()
                      if module is find_cam_target_layer(model))
    default_maps = experiment.manager.get_explainer_by_id(default_id).attribute(inputs, targets)
    default_diff = (default_maps.cpu() - maps).abs().max().item()
    print("[8] recommended:", UCAG in experiment.recommended.explainers, "| default layer", discovered,
          "| max abs diff to layer4", default_diff)

    # DOC: optimization
    optimized = experiment.optimize(
        data_ids=tuning_ids, explainer_id=uid, metric_id=metric_ids[0],
        direction='maximize', sampler='random', n_trials=args.n_trials, seed=seed)
    # Optuna returns deep copies; metrics require the experiment's model object.
    selected_id = experiment.manager.add_explainer(experiment.manager.get_explainer_by_id(uid).set_kwargs(
        alpha=optimized.explainer.alpha, num_units=optimized.explainer.num_units))
    selected_pid = experiment.manager.add_postprocessor(optimized.postprocessor)
    # END DOC: optimization
    trials = [{
        "number": trial.number,
        "alpha": trial.params["explainer.alpha"],
        "num_units": trial.params["explainer.num_units"],
        "pooling": trial.params["postprocessor.pooling_fn.method"],
        "normalization": trial.params["postprocessor.normalization_fn.method"],
        "objective_sum": trial.value,
        "state": trial.state.name,
    } for trial in optimized.study.trials]
    for trial in trials:
        print("[7] trial", trial)

    selected_scores = {}
    for metric_id in metric_ids:
        set_seed(seed)
        selected = experiment.run_batch(explainer_id=selected_id, postprocessor_id=selected_pid,
                                        metric_id=metric_id, data_ids=final_ids)
        selected_scores[type(selected["metric"]).__name__] = selected["evaluation"].tolist()
    save_figure(inputs, display(selected["postprocessed"]), titles, output_dir / "optimized-ucag.png")

    explainer = experiment.manager.get_explainer_by_id(uid)
    explainer.attribute(inputs[:1], targets[:1])  # warm-up
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(3):
        synchronize(device)
        started = time.perf_counter()
        explainer.attribute(inputs[:1], targets[:1])
        synchronize(device)
        durations.append(time.perf_counter() - started)

    mean = lambda values: sum(values) / len(values)
    record = {
        "model": "resnet50", "weights": "IMAGENET1K_V1",
        "preprocessing": "Resize((224, 224)), ToTensor, ImageNet Normalize", "seed": seed,
        "final_ids": FINAL_IDS, "tuning_ids": TUNING_IDS,
        "final_files": [dataset.dataset.img_names[i] for i in FINAL_IDS],
        "tuning_files": [dataset.dataset.img_names[i] for i in TUNING_IDS],
        "targets": "ground-truth labels", "final_targets": targets.tolist(),
        "final_target_names": target_names, "final_predictions": predictions,
        "final_prediction_names": prediction_names,
        "explainer": {name: getattr(explainer, name) for name in
                      ("layer", "crop_size", "num_units", "alpha", "limit_batch_size", "score")},
        "maps": {"shape": list(maps.shape), "dtype": str(maps.dtype),
                 "min": maps.min().item(), "max": maps.max().item()},
        "recommended_by_autoexplanation": UCAG in experiment.recommended.explainers,
        "recommended_instance": {"discovered_layer": discovered, "max_abs_diff_to_configured": default_diff},
        "baseline": {"postprocessor": "sumpos/identity", "scores": scores,
                     "means": {name: mean(values) for name, values in scores.items()}},
        "trials": trials, "metric_perturbations": perturbations,
        "selected": {"alpha": optimized.explainer.alpha, "num_units": optimized.explainer.num_units,
                     "postprocessor": repr(optimized.postprocessor), "scores": selected_scores,
                     "means": {name: mean(values) for name, values in selected_scores.items()}},
        "timing": {"explanation_s_per_image": mean(durations),
                   "explanation_peak_cuda_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                 if device.type == "cuda" else None),
                   "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "captum": captum.__version__,
                        "optuna": optuna.__version__, "device": device_name(device),
                        "threads": torch.get_num_threads()},
    }
    (output_dir / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print("[6] baseline means", record["baseline"]["means"], "| selected means", record["selected"]["means"])
    print("[9] explanation s/image", round(mean(durations), 3), "| peak RSS GiB",
          round(record["timing"]["process_peak_rss_gib"], 2))
    print("[done]", output_dir / "results.json")


if __name__ == "__main__":
    main()

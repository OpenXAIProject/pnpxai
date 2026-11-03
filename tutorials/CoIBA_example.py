"""Runnable companion to tutorials/CoIBA_example.md.

Run from the repository root after installing pnpxai:

    python tutorials/CoIBA_example.py --model resnet50
    python tutorials/CoIBA_example.py --model vit_b_16

The 1000 ImageNet sample images in tutorials/data/ImageNet are split into 992
calibration, 4 tuning and 4 final images. Figures, results.json and a reusable
statistics cache (statistics.pt, gitignored) are written to
tutorials/data/coiba/<model>/. Python snippets in the Markdown are copied from
the DOC blocks below.
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
from torch.utils.data import DataLoader, Subset, TensorDataset

from pnpxai import AutoExplanationForImageClassification
from pnpxai.evaluator.metrics import Complexity, MuFidelity
from pnpxai.explainers import CoIBA
from pnpxai.explainers.utils.postprocess import Identity, PostProcessor, SumPos
from pnpxai.utils import set_seed

from helpers import denormalize_image, get_imagenet_dataset, get_torchvision_model

# model -> (layers, feature layout, prefix tokens kept out of the bottleneck)
CONFIGS = {
    "resnet50": (("layer3", "layer4"), "NCHW", 0),
    "vit_b_16": (("encoder.layers.encoder_layer_4", "encoder.layers.encoder_layer_5"), "NTC", 1),
}
FINAL_IDS = [0, 7, 14, 403]
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


def stack(dataset, ids, device):
    images, labels = zip(*(dataset[i] for i in ids))
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


def display(maps):
    """Per-image min-max for display; constant maps stay zero."""
    flat = maps.flatten(1)
    low, high = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
    return ((flat - low) / (high - low).clamp_min(1e-12)).view_as(maps)


def save_figure(images, maps, titles, transform, path):
    count = len(images)
    fig, axes = plt.subplots(2, count, figsize=(2.0 * count, 4.2), squeeze=False)
    for i in range(count):
        image = denormalize_image(images[i].cpu(), transform.mean, transform.std).clip(0, 1)
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
    parser.add_argument("--model", choices=sorted(CONFIGS), default="resnet50")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--estimate-samples", type=int, default=992)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--perturbations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    # get_imagenet_dataset() changes the working directory, so resolve first.
    output_dir = (args.output_dir or Path(__file__).resolve().parent / "data" / "coiba" / args.model).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "statistics.pt"

    seed, perturbations = args.seed, args.perturbations
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, transform = get_torchvision_model(args.model)
    model = model.to(device).eval()
    weights = torchvision.models.get_model_weights(args.model).DEFAULT.name
    preprocessing_id = f"{args.model}-{weights}-default-transforms"
    layers, layout, prefix = CONFIGS[args.model]

    dataset = get_imagenet_dataset(transform, subset_size=1000)
    calibration_ids = [i for i in range(len(dataset)) if i not in FINAL_IDS + TUNING_IDS]
    calibration_loader = DataLoader(Subset(dataset, calibration_ids), batch_size=32)
    images, labels = stack(dataset, FINAL_IDS, device)
    tuning_images, tuning_labels = stack(dataset, TUNING_IDS, device)
    with torch.no_grad():
        targets = model(images).argmax(-1)
    names = [dataset.dataset.idx_to_label(t) for t in targets.tolist()]
    print("[3] model / layers / targets", args.model, layers, targets.tolist(), names)

    # DOC: explicit_constructor
    explainer = CoIBA(
        model=model, layer=layers, feature_layout=layout, skip_prefix_tokens=prefix,
        beta=10, optimization_steps=10, lr=1, batch_size=10, min_std=.01,
        seed=seed, forward_arg_extractor=None, additional_forward_arg_extractor=None,
    )
    # END DOC: explicit_constructor

    synchronize(device)
    started = time.perf_counter()
    # DOC: preparation
    explainer.estimate(calibration_loader, n_samples=args.estimate_samples)
    explainer.save_statistics(cache_path, preprocessing_id=preprocessing_id)
    # END DOC: preparation
    synchronize(device)
    preparation_s = time.perf_counter() - started
    restored = CoIBA(model=model, layer=layers, feature_layout=layout,
                     skip_prefix_tokens=prefix, seed=seed)
    restored.load_statistics(cache_path, preprocessing_id=preprocessing_id)

    # DOC: attribute
    maps = explainer.attribute(images, targets)
    torch.testing.assert_close(restored.attribute(images, targets), maps)
    assert maps.shape == (len(images), 1, *images.shape[-2:])
    assert maps.dtype == images.dtype and maps.device == images.device
    assert maps.isfinite().all() and maps.min() >= 0 and maps.max() <= 1
    # END DOC: attribute
    print("[4] maps", tuple(maps.shape), maps.dtype, maps.device, "range", maps.min().item(), maps.max().item())

    # DOC: postprocessing
    postprocessor = PostProcessor.from_name(
        pooling_method='sumpos', normalization_method='identity', channel_dim=1)
    pooled = postprocessor(maps)
    assert pooled.shape == (len(images), *images.shape[-2:])
    # END DOC: postprocessing
    save_figure(images, display(pooled), names, transform, output_dir / "attributions.png")

    # DOC: metrics
    metrics = [MuFidelity(model, explainer, n_perturb=perturbations, batch_size=4),
               Complexity(model, explainer, n_bins=10)]
    # END DOC: metrics
    baseline = {}
    for metric in metrics:
        set_seed(seed)
        baseline[type(metric).__name__] = metric.evaluate(images, targets, pooled).tolist()

    combined_loader = DataLoader(TensorDataset(torch.cat([tuning_images, images]),
                                               torch.cat([tuning_labels, labels])), batch_size=4)
    tuning_ids, final_ids = [0, 1, 2, 3], [4, 5, 6, 7]
    # DOC: automatic_pipeline
    expr = AutoExplanationForImageClassification(
        model, combined_loader, input_extractor=lambda batch: batch[0],
        label_extractor=lambda batch: batch[1], target_extractor=lambda logits: logits.argmax(-1),
        target_labels=False)
    eid = expr.manager.add_explainer(explainer)
    mufidelity_id = expr.manager.add_metric(metrics[0])
    # END DOC: automatic_pipeline
    postprocessor_id = next(
        i for i, pp in enumerate(expr.manager.postprocessors)
        if isinstance(pp.pooling_fn, SumPos) and isinstance(pp.normalization_fn, Identity))
    set_seed(seed)
    result = expr.run_batch(explainer_id=eid, postprocessor_id=postprocessor_id,
                            metric_id=mufidelity_id, data_ids=final_ids)
    torch.testing.assert_close(result["explanation"], maps.cpu())  # the cache keeps CPU copies
    print("[8] recommended:", CoIBA in expr.recommended.explainers, "| run_batch",
          tuple(result["explanation"].shape), tuple(result["postprocessed"].shape), tuple(result["evaluation"].shape))

    expr.predict_batch(tuning_ids)  # optimize() reads cached outputs for targets
    # DOC: optimization
    optimized = expr.optimize(
        data_ids=tuning_ids, explainer_id=eid, metric_id=mufidelity_id,
        direction='maximize', sampler='random', n_trials=args.n_trials, seed=seed)
    # END DOC: optimization
    trials = [{
        "number": trial.number,
        "beta": trial.params["explainer.beta"],
        "pooling": trial.params["postprocessor.pooling_fn.method"],
        "normalization": trial.params["postprocessor.normalization_fn.method"],
        "objective_sum": trial.value,
        "state": trial.state.name,
    } for trial in optimized.study.trials]
    for trial in trials:
        print("[7] trial", trial)

    selected = explainer.set_kwargs(beta=optimized.explainer.beta)
    selected_maps = selected.attribute(images, targets)
    selected_pooled = optimized.postprocessor(selected_maps)
    selected_scores = {}
    for metric in (MuFidelity(model, selected, n_perturb=perturbations, batch_size=4),
                   Complexity(model, selected, n_bins=10)):
        set_seed(seed)
        selected_scores[type(metric).__name__] = metric.evaluate(images, targets, selected_pooled).tolist()
    save_figure(images, display(selected_pooled), names, transform, output_dir / "optimized-attributions.png")

    explainer.attribute(images[:1], targets[:1])  # warm-up
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(3):
        synchronize(device)
        started = time.perf_counter()
        explainer.attribute(images[:1], targets[:1])
        synchronize(device)
        durations.append(time.perf_counter() - started)

    mean = lambda values: sum(values) / len(values)
    results = {
        "model": args.model, "weights": weights, "layers": list(layers), "feature_layout": layout,
        "skip_prefix_tokens": prefix, "seed": seed, "preprocessing_id": preprocessing_id,
        "calibration": {"ids": "all dataset ids except final_ids and tuning_ids",
                        "n_samples": min(args.estimate_samples, len(calibration_ids))},
        "final_ids": FINAL_IDS, "tuning_ids": TUNING_IDS,
        "final_files": [dataset.dataset.img_names[i] for i in FINAL_IDS],
        "tuning_files": [dataset.dataset.img_names[i] for i in TUNING_IDS],
        "final_targets": targets.tolist(), "final_target_names": names,
        "maps": {"shape": list(maps.shape), "dtype": str(maps.dtype),
                 "min": maps.min().item(), "max": maps.max().item()},
        "baseline": {"beta": explainer.beta, "postprocessor": "sumpos/identity", "scores": baseline,
                     "means": {name: mean(values) for name, values in baseline.items()}},
        "selected": {"beta": selected.beta, "postprocessor": repr(optimized.postprocessor),
                     "scores": selected_scores,
                     "means": {name: mean(values) for name, values in selected_scores.items()}},
        "trials": trials, "metric_perturbations": perturbations,
        "recommended_by_autoexplanation": CoIBA in expr.recommended.explainers,
        "timing": {"preparation_s": preparation_s, "explanation_s_per_image": mean(durations),
                   "explanation_peak_cuda_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                 if device.type == "cuda" else None),
                   "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "captum": captum.__version__,
                        "optuna": optuna.__version__, "device": device_name(device),
                        "threads": torch.get_num_threads()},
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("[6] baseline means", results["baseline"]["means"], "| selected means", results["selected"]["means"])
    print("[9] preparation s", round(preparation_s, 3), "| explanation s/image", round(mean(durations), 3))
    print("[done]", output_dir / "results.json")


if __name__ == "__main__":
    main()

"""Runnable companion to tutorials/attribution_equilibrium.md.

Run from the repository root after installing pnpxai:

    python tutorials/attribution_equilibrium.py
    python tutorials/attribution_equilibrium.py --device cpu --n-trials 2
    python tutorials/attribution_equilibrium.py --bench --device cpu

Sections 3-8 run by default. The first run downloads ResNet18 weights if they
are not cached. ImageNet sample images are read from tutorials/data/ImageNet.
Python snippets in the Markdown are copied from the DOC blocks below.
"""

import argparse
from pathlib import Path
import time

# DOC: imports
import torch
from torch.utils.data import DataLoader

from pnpxai import AutoExplanationForImageClassification
from pnpxai.core.modality import ImageModality
from pnpxai.evaluator.metrics import AbPC, Complexity
from pnpxai.explainers import AE
from pnpxai.explainers.utils.postprocess import (
    Identity, L1Norm, MinMax, PoolingFunction, PostProcessor,
)
from pnpxai.utils import set_seed
from helpers import (
    denormalize_image, get_imagenet_dataset, get_torchvision_model,
)
# END DOC: imports


# --- 3. Model and data -------------------------------------------------------
# DOC: setup
def setup(batch_size=1, device_name="auto"):
    set_seed(0)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    model, transform = get_torchvision_model("resnet18")
    model = model.to(device).eval()

    dataset = get_imagenet_dataset(transform=transform, indices=[0, 7, 14, 403])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        targets = model(inputs).argmax(-1)
    print("[3] device / images / targets", device, tuple(inputs.shape), targets.tolist())
    return model, transform, dataset, loader, inputs, labels, targets, device
# END DOC: setup


# --- 4. Explain --------------------------------------------------------------
# DOC: explain
def explain(model, inputs, targets, layer="layer4", ratio=2.0):
    explainer = AE(model=model, layer=layer, ratio=ratio)
    attrs = explainer.attribute(inputs, targets)
    print("[4] shape / dtype / device", tuple(attrs.shape), attrs.dtype, attrs.device)
    print("[4] negative / positive", bool((attrs < 0).any()), bool((attrs > 0).any()))
    print("[4] range", attrs.min().item(), attrs.max().item())
    assert attrs.shape == inputs.shape
    assert attrs.dtype == inputs.dtype
    assert attrs.device == inputs.device
    assert torch.isfinite(attrs).all()
    return explainer, attrs
# END DOC: explain


# --- 5. Post-process and visualize -------------------------------------------
# DOC: postprocess
class SignedSum(PoolingFunction):
    """Keep both signs while summing RGB channels for the original AE view."""

    def __call__(self, attrs):
        return attrs.sum(dim=self.channel_dim)


def postprocess(attrs):
    signed_processor = PostProcessor(
        pooling_fn=SignedSum(channel_dim=1), normalization_fn=Identity(),
    )
    metric_processor = PostProcessor.from_name(
        pooling_method="l1norm", normalization_method="minmax", channel_dim=1,
    )
    signed = signed_processor(attrs)
    pooled = metric_processor(attrs)
    print("[5] raw -> signed / metric", tuple(attrs.shape), tuple(signed.shape), tuple(pooled.shape))
    print("[5] metric range", pooled.min().item(), pooled.max().item())
    assert signed.shape == pooled.shape == attrs.shape[:1] + attrs.shape[2:]
    assert torch.isfinite(pooled).all(), "Constant maps need identity normalization."
    return signed, pooled
# END DOC: postprocess


# DOC: visualize
def visualize(inputs, labels, signed, pooled, transform, dataset, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(inputs.shape[0], 3, figsize=(9.6, 3.2 * inputs.shape[0]), squeeze=False)
    for index in range(inputs.shape[0]):
        image = denormalize_image(inputs[index].detach().cpu(), transform.mean, transform.std)
        axes[index, 0].imshow(image.clip(0, 1))
        axes[index, 0].set_title(dataset.dataset.idx_to_label(labels[index].item()))
        limit = max(signed[index].abs().max().item(), 1e-12)
        axes[index, 1].imshow(signed[index].cpu().numpy(), cmap="seismic", vmin=-limit, vmax=limit)
        axes[index, 1].set_title("AE signed RGB sum")
        axes[index, 2].imshow(pooled[index].cpu().numpy(), cmap="magma", vmin=0, vmax=1)
        axes[index, 2].set_title("Magnitude map for metrics")
        for axis in axes[index]:
            axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("[5] wrote", path)
# END DOC: visualize


# --- 6. Evaluate -------------------------------------------------------------
# DOC: evaluate
def evaluate(model, explainer, inputs, targets, pooled):
    # Use post-processed maps, matching Experiment.evaluate_batch.
    metrics = [
        AbPC(model=model, explainer=explainer, channel_dim=1, n_steps=10),
        Complexity(model=model, explainer=explainer, n_bins=10),
    ]
    scores_by_name = {}
    for metric in metrics:
        scores = metric.evaluate(inputs, targets, pooled)
        name = type(metric).__name__
        print(f"[6] {name:12s} shape={tuple(scores.shape)} mean={scores.mean().item():.6f}")
        assert scores.shape == (inputs.shape[0],)
        assert torch.isfinite(scores).all()
        scores_by_name[name] = scores.detach().cpu()
    return scores_by_name
# END DOC: evaluate


# --- 7. Optimize -------------------------------------------------------------
# DOC: experiment
def make_auto_experiment(model, loader, device):
    expr = AutoExplanationForImageClassification(
        model=model,
        data=loader,
        input_extractor=lambda batch: batch[0].to(device),
        label_extractor=lambda batch: batch[-1].to(device),
        target_extractor=lambda outputs: outputs.argmax(-1),
        target_labels=False,
    )
    # Find ids by type: adding another explainer changes the sorted ids.
    ae_id = next(i for i, e in enumerate(expr.manager.explainers) if isinstance(e, AE))
    abpc_id = next(i for i, m in enumerate(expr.manager.metrics) if isinstance(m, AbPC))
    postprocessor_id = next(
        i for i, pp in enumerate(expr.manager.postprocessors)
        if isinstance(pp.pooling_fn, L1Norm) and isinstance(pp.normalization_fn, MinMax)
    )
    return expr, ae_id, abpc_id, postprocessor_id
# END DOC: experiment


# DOC: optimize
def optimize(expr, ae_id, abpc_id, n_trials=5):
    # optimize() reads cached predictions when deriving the target classes.
    expr.predict_batch(data_ids=[0])
    optimized = expr.optimize(
        data_ids=0,
        explainer_id=ae_id,
        metric_id=abpc_id,
        direction="maximize",  # AbPC: higher is better.
        sampler="tpe",
        n_trials=n_trials,
        seed=42,
    )
    print("[7] best ratio", optimized.explainer.ratio)
    print("[7] best postprocessor", optimized.postprocessor)
    print("[7] best AbPC", optimized.study.best_trial.value)
    assert 1 <= optimized.explainer.ratio <= 2
    return optimized
# END DOC: optimize


# --- 8. Through AutoExplanation ---------------------------------------------
# DOC: auto
def run_auto(expr, ae_id, abpc_id, postprocessor_id, inputs):
    expr.recommended.print_tabular()
    automatic = expr.manager.explainers[ae_id]
    print("[8] AE id / layer / ratio", ae_id, automatic.layer, automatic.ratio)
    print("[8] baseline / feature-mask attributes",
          hasattr(automatic, "baseline_fn"), hasattr(automatic, "feature_mask_fn"))
    results = expr.run_batch(
        explainer_id=ae_id,
        postprocessor_id=postprocessor_id,
        metric_id=abpc_id,
        data_ids=range(inputs.shape[0]),
    )
    for key in ("explanation", "postprocessed", "evaluation"):
        print("[8]", key, tuple(results[key].shape))
        assert torch.isfinite(results[key]).all()
    assert results["explanation"].shape == inputs.shape
    assert results["postprocessed"].shape == inputs.shape[:1] + inputs.shape[2:]
    assert results["evaluation"].shape == (inputs.shape[0],)
    return results
# END DOC: auto


def bench(model, inputs, targets, layer="layer4", ratio=2.0, repeats=3):
    explainer = AE(model=model, layer=layer, ratio=ratio)
    explainer.attribute(inputs, targets)
    if inputs.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        explainer.attribute(inputs, targets)
    if inputs.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / (repeats * inputs.shape[0])
    peak = (f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak allocated"
            if inputs.is_cuda else "CPU peak memory not measured")
    print(f"[bench] {elapsed * 1000:.1f} ms/sample; {peak}; batch={inputs.shape[0]}, ratio={ratio}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--layer", default="layer4")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "result" / "attribution_equilibrium.png",
    )
    args = parser.parse_args()
    if not 1 <= args.ratio <= 2 or min(args.n_trials, args.repeats, args.threads) < 1:
        parser.error("ratio must be in [1, 2]; trials, repeats, and threads must be positive")
    torch.set_num_threads(args.threads)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, transform, dataset, loader, inputs, labels, targets, device = setup(
        batch_size=args.batch_size, device_name=args.device,
    )
    if args.bench:
        bench(model, inputs, targets, args.layer, args.ratio, args.repeats)
        return
    explainer, attrs = explain(model, inputs, targets, args.layer, args.ratio)
    signed, pooled = postprocess(attrs)
    if not args.no_plot:
        visualize(inputs, labels, signed, pooled, transform, dataset, output_path)
    evaluate(model, explainer, inputs, targets, pooled)
    expr, ae_id, abpc_id, pp_id = make_auto_experiment(model, loader, device)
    optimize(expr, ae_id, abpc_id, n_trials=args.n_trials)
    run_auto(expr, ae_id, abpc_id, pp_id, inputs)
    print(f"[done] elapsed={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()

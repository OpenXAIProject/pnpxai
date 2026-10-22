"""
Runnable source for docs/tutorials/mad_xai.md

Repo destination: tutorials/mad_xai.py
Run from the tutorials/ directory so that helpers.py resolves:

    python mad_xai.py            # sections 3-8 of the tutorial
    python mad_xai.py --bench    # fills the card's "Cost" row

Every code block in the tutorial markdown is copied from this file. If you
change one, change the other.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pnpxai.evaluator.metrics import Complexity
from pnpxai.explainers import MadXai
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.utils import set_seed

from helpers import get_brats_dataset


# --- 3. Model and data -------------------------------------------------------

def setup(batch_size=8):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 126 axial T2 slices (one per subject) with brain masks and lesion masks.
    dataset = get_brats_dataset(modality="t2", image_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Fixed subject-level split: settings are chosen on `dev`, reported on `heldout`.
    perm = np.random.RandomState(0).permutation(len(dataset))
    dev = np.zeros(len(dataset), dtype=bool)
    dev[perm[: len(dataset) // 2]] = True
    return device, dataset, loader, dev


# --- 4. Explain --------------------------------------------------------------

def explain(device, loader):
    explainer = MadXai(
        ddim_steps=10,               # default: DDIM steps per reverse pass
        n_samples=32,                # default: stochastic normal counterparts per input
        eta=1.0,                     # default: DDIM stochasticity
        mask_threshold=0.9,          # default: regenerate patches with p(normal) < 0.9
        n_iters=2,                   # default: reverse passes, each from the previous output
        normalization="uncertainty", # default: residual / (sample std + eps)
        sign="abs",                  # default: any deviation counts
        eps=0.05,                    # default
        clip=2.0,                    # default: scores clipped to [0, 2], then scaled to [0, 1]
        sigma=5.0,                   # default: final gaussian smoothing (pixels)
        seed=0,
        model_device=device,
    )
    inputs, brain_mask, seg = next(iter(loader))
    attrs = explainer.attribute(inputs)

    # These must match the "Output" row of docs/_cards/mad_xai.md.
    print("[4] inputs.shape         ", inputs.shape)
    print("[4] attrs.shape          ", attrs.shape)
    print("[4] attrs.dtype          ", attrs.dtype)
    print("[4] same device as input ", attrs.device == inputs.device)
    print("[4] range                ", attrs.min().item(), attrs.max().item())
    assert attrs.shape == (inputs.shape[0], *inputs.shape[-2:])
    assert attrs.device == inputs.device
    assert attrs.min() >= 0 and attrs.max() <= 1
    return explainer, inputs, seg, attrs


# --- 5. Post-process and visualize -------------------------------------------

def postprocess(explainer, inputs, seg, attrs, path="mad_xai_example.png"):
    # The map is already one channel and in [0, 1]: identity pooling and normalization.
    postprocessor = PostProcessor.from_name(
        pooling_method="identity", normalization_method="identity", channel_dim=1,
    )
    pooled = postprocessor(attrs)
    print("[5] before / after       ", tuple(attrs.shape), tuple(pooled.shape))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, counterparts = explainer.reconstruct(inputs[:4])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        cols = [inputs[i, 0], counterparts[i].mean((0, 1)).cpu(), pooled[i], seg[i]]
        titles = ["input (T2)", "normal counterpart (mean)", "MadXai map", "lesion (GT)"]
        for j, (img, title) in enumerate(zip(cols, titles)):
            axes[i, j].imshow(img.numpy(), cmap="jet" if j == 2 else "gray")
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    print("[5] saved                ", path)
    return pooled


# --- 6. Evaluate -------------------------------------------------------------

THRESHOLDS = np.linspace(0, 1, 201)


def dice_per_threshold(maps, segs):
    """(T, N) Dice for every threshold and image. Same rules as dsc_max() in the
    MAD-AD evaluation script: empty ground truth scores 1 if the prediction is empty."""
    m, s = maps.flatten(1).float(), segs.flatten(1).float()
    s_sum = s.sum(1)
    out = []
    for th in THRESHOLDS:
        b = (m > th).float()
        b_sum = b.sum(1)
        d = 2 * (b * s).sum(1) / (b_sum + s_sum + 1e-6)
        out.append(torch.where(s_sum == 0, (b_sum == 0).float(), d))
    return torch.stack(out).numpy()


def explain_all(explainer, loader):
    maps, segs = [], []
    for inputs, brain_mask, seg in loader:
        maps.append(explainer.attribute(inputs))
        segs.append(seg)
    return torch.cat(maps), torch.cat(segs)


def evaluate(explainer, loader, dev):
    from sklearn.metrics import average_precision_score, roc_auc_score
    maps, segs = explain_all(explainer, loader)
    D = dice_per_threshold(maps, segs)
    th = int(np.argmax(D[:, dev].mean(1)))          # threshold chosen on dev only
    y, p = segs.flatten().numpy(), maps.flatten().numpy()
    # Complexity only reads the attributions; the metric still needs a module for its device.
    complexity = Complexity(model=explainer.unet_model, explainer=explainer)
    result = {
        "max-Dice (all 126)": D.mean(1).max(),
        "Dice heldout @ dev threshold": D[th, ~dev].mean(),
        "dev threshold": THRESHOLDS[th],
        "pixel AUROC": roc_auc_score(y, p),
        "pixel AP": average_precision_score(y, p),
        "Complexity (entropy, lower = sparser)": complexity.evaluate(None, None, attributions=maps).mean().item(),
    }
    for k, v in result.items():
        print(f"[6] {k:38s} {v:.4f}")
    return result


# --- 7. Optimize -------------------------------------------------------------

def optimize(device, loader, dev, dataset):
    # experiment.optimize needs a classifier-based metric, which MadXai does not have.
    # Instead, compare mask_threshold values on the dev split with max-Dice.
    dev_loader = DataLoader(torch.utils.data.Subset(dataset, np.where(dev)[0]),
                            batch_size=8, shuffle=False)
    for mask_threshold in [0.5, 0.9]:
        explainer = MadXai(mask_threshold=mask_threshold, seed=0, model_device=device)
        maps, segs = explain_all(explainer, dev_loader)
        print(f"[7] mask_threshold={mask_threshold}: dev max-Dice "
              f"{dice_per_threshold(maps, segs).mean(1).max():.4f}")


# --- 8. Through AutoExplanation ----------------------------------------------

def auto_explanation():
    # MadXai explains brain MRI with its own diffusion model, not a classifier's
    # modules, so the recommender has nothing to match it against.
    print("[8] SUPPORTED_MODULES    ", MadXai.SUPPORTED_MODULES)


# --- Cost --------------------------------------------------------------------

def bench(device, loader):
    explainer = MadXai(seed=0, model_device=device)
    inputs, _, _ = next(iter(loader))
    explainer.attribute(inputs[:1])                  # warm-up
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    explainer.attribute(inputs)
    torch.cuda.synchronize()
    per_slice = (time.time() - start) / len(inputs)
    print(f"[bench] {torch.cuda.get_device_name()}  batch={len(inputs)}  "
          f"{per_slice:.2f} s/slice  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    device, dataset, loader, dev = setup()
    if args.bench:
        bench(device, loader)
    else:
        explainer, inputs, seg, attrs = explain(device, loader)
        postprocess(explainer, inputs, seg, attrs)
        # A fresh explainer so the noise stream starts at the first slice.
        evaluate(MadXai(seed=0, model_device=device), loader, dev)
        print("[6] legacy configuration (pnpxai 0.1.x):")
        evaluate(MadXai(**MadXai.legacy_config(), seed=0, model_device=device), loader, dev)
        optimize(device, loader, dev, dataset)
        auto_explanation()

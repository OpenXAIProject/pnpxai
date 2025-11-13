import os
import argparse
import itertools
import re
import random
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- OmniXAI Imports ---
try:
    from omnixai.explainers.vision import (
        IntegratedGradientImage,
        GradCAM,
        LimeImage,
        # ShapImage,
        SmoothGrad,
    )
except ImportError as e:
    print(f"Error importing OmniXAI: {e}")
    exit()

# --- Captum Imports ---
from captum.attr._utils.visualization import _normalize_attr as normalize_attr

# --- pnpxai Metric Imports ---
try:
    from pnpxai.evaluator.metrics import AbPC, Complexity, Simplicity, CompositeMetrics
except ImportError as e:
    print(f"Error importing pnpxai metrics: {e}")
    exit()

# --- Helper Imports ---
try:
    from tutorials.helpers import get_torchvision_model, get_imagenet_val_dataset, denormalize_image, save_pickle_data, load_pickle_data
    from pnpxai.utils import set_seed
    from pnpxai.explainers.utils import find_cam_target_layer
except ImportError as e:
    print(f"Error importing helper functions: {e}")

import pdb


TORCHVISION_MODEL_CHOICES = [
    'resnet18',
    'vit_b_16',
    # ...
]


# --- Evaluation Function using OmniXAI ---
def evaluate(args):
    # setup
    cwd = os.getcwd()
    savedir_base = os.path.join(cwd, f'results/omnixai/{args.model}')
    os.makedirs(savedir_base, exist_ok=True)
    os.makedirs(os.path.join(savedir_base, 'evals'), exist_ok=True)

    use_gpu = torch.cuda.is_available() and not args.disable_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')
    set_seed(0)
    print(f"Using device: {device}")

    # prepare model
    print(f"Loading model: {args.model}")
    model, transform = get_torchvision_model(args.model)
    model.to(device)
    model.eval()
    dtype = next(model.parameters()).dtype

    # prepare data
    # print(f"Loading ImageNet validation data from: {args.data_dir}")
    dataset = get_imagenet_val_dataset(transform, '/data')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        shuffle=False,
    )
    print(f"Dataset size: {len(dataset)}, DataLoader size: {len(dataloader)}")

    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Check dataset path or size.")

    # --- Define OmniXAI Explainers ---
    cam_target_layer = find_cam_target_layer(model)

    # OmniXAI requires a predictor function
    def predict_function(images):
        # Assuming images are ndarray: (Batch, 224, 224, 3)
        images = images.transpose(0, 3, 1, 2)
        inputs = torch.tensor(images, device=device, dtype=dtype)
        logits = model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    # OmniXAI requires a preprocess function
    preprocess = lambda x: x.permute(0, 3, 1, 2) if isinstance(x, torch.Tensor) else x.transpose(0, 3, 1, 2)

    omnixai_explainers = {
        'gradient': None,
        'gradient_x_input': None,
        'smooth_grad': SmoothGrad(model=model, preprocess_function=preprocess),
        'var_grad': None,
        'lrp': None,
        'grad_cam': GradCAM(model=model, target_layer=cam_target_layer, preprocess_function=preprocess),
        'guided_grad_cam': None,
        # 'kernel_shap': ShapImage(model),
        # 'lime': LimeImage(predict_function=model),
        'lime': LimeImage(predict_function=predict_function),
        'integrated_gradients': IntegratedGradientImage(model=model, preprocess_function=preprocess),
        'rap': None,
    }

    # Filter out None explainers
    active_explainers = defaultdict(dict)
    for explainer in args.eval_explainer:
        if (explainer in omnixai_explainers) and (omnixai_explainers[explainer] is not None):
             print(f"Evaluating explainer: {explainer}")
             active_explainers[explainer] = omnixai_explainers[explainer]
        else:
            print(f"Requested explainer '{explainer}' not available or inactive. Evaluating all active ones.")

    print(f"Active explainers for evaluation: {list(active_explainers.keys())}")
    if 'rap' not in omnixai_explainers:
         print("Note: RAP explainer is not available in Captum and will be skipped.")
    if cam_target_layer is None and ('grad_cam' in omnixai_explainers or 'guided_grad_cam' in omnixai_explainers):
        print("Warning: Could not find target layer. GradCAM and GuidedGradCAM will be skipped.")

    # --- Instantiate pnpxai Metrics ---
    print("Instantiating metrics...")
    try:
        # Metrics instantiation remains the same as it uses pnpxai
        abpc_metric = AbPC(model=model, target_input_keys=[0]) # Assuming target_input_keys=[0] is correct for image input

        simplicity_metric = Simplicity(model=model)
        complexity_metric = Complexity(model=model)

        composite_metric = CompositeMetrics(model=model, target_input_keys=[0])
        composite_metric.composite_metrics = [abpc_metric, simplicity_metric]
        composite_metric.composite_weights = torch.tensor([0.7, 0.3])

        metrics_to_run = {
            'ab_pc': abpc_metric,
            'complexity': complexity_metric,
            'composite': composite_metric,
        }
        print("Metrics instantiated successfully.")
    except Exception as e:
        print(f"Error during metric instantiation: {e}")
        print("Check metric class definitions and required arguments in pnpxai files.")
        exit()


    # --- Evaluation Loop ---
    print("\n--- Starting OmniXAI Evaluation ---")
    results = defaultdict(lambda: defaultdict(list)) # results[explainer][metric] = [scores]

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating Batches")):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad_()

        # Get model prediction (needed for target index) - use no_grad here
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            pred_label_idx = torch.argmax(outputs, dim=1)

        for explainer_name, explainer_instance in active_explainers.items():
            print(explainer_name)
            tqdm.write(f"  Batch {batch_idx+1}/{len(dataloader)}, Explainer: {explainer_name}")

            # --- Compute Attributions ---
            current_attribution = None
            if isinstance(explainer_instance, GradCAM):
                expls = explainer_instance.explain(X=inputs.permute(0, 2, 3, 1), y=pred_label_idx)
                current_attribution = np.stack(
                    [expl['scores'] for expl in expls.get_explanations()]
                )[:, :, :, None]

            elif isinstance(explainer_instance, LimeImage):
                expls = explainer_instance.explain(X=inputs.permute(0, 2, 3, 1))
                current_attribution = np.stack(
                    [expl['masks'][0] for expl in expls.get_explanations()]
                )[:, :, :, None]

            elif isinstance(explainer_instance, IntegratedGradientImage):
                expls = explainer_instance.explain(X=inputs.permute(0, 2, 3, 1), y=pred_label_idx)
                current_attribution = np.stack(
                    [expl['scores'] for expl in expls.get_explanations()]
                )

            elif isinstance(explainer_instance, SmoothGrad):
                expls = explainer_instance.explain(X=inputs.permute(0, 2, 3, 1), y=pred_label_idx)
                current_attribution = np.stack(
                    [expl['scores'] for expl in expls.get_explanations()]
                )

            else:
                continue

            # --- Postprocessing ---
            if isinstance(explainer_instance, LimeImage):
                pp_attribution = (current_attribution + 1) * 0.5
            else:
                pp_attribution = normalize_attr(
                    current_attribution, sign='absolute_value', reduction_axis=-1
                )
            pp_attribution = torch.tensor(pp_attribution)

            ''' Savefigs
            for i in range(2):
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
                axs[0].imshow(denormalize_image(inputs[i].detach().cpu(), mean=transform.mean, std=transform.std))
                axs[1].imshow(pp_attribution[i], cmap='Reds')
                [ax.axis('off') for ax in axs]
                plt.tight_layout()
                plt.savefig(f'omnixai_attrs_debug/smooth_grad/omnixai_smooth_grad_{batch_idx * 2 + i}.jpg')
                plt.close()

            # pdb.set_trace()
            '''

            # --- Calculate Metrics ---
            metric_scores = {}
            for metric_name, metric_instance in metrics_to_run.items():
                if metric_name == 'complexity':
                    score = metric_instance.evaluate(inputs=inputs, attributions=pp_attribution, targets=pred_label_idx)
                    # print(f'complexity score: {score}')
                elif metric_name == 'ab_pc' or metric_name == 'simplicity': # Assuming Simplicity is like AbPC
                    score = metric_instance.evaluate(inputs=inputs, attributions=pp_attribution, targets=pred_label_idx)
                    # print(f'abpc score: {score}')
                elif metric_name == 'composite':
                    score = metric_instance.evaluate(inputs=inputs, attributions=pp_attribution, targets=pred_label_idx)
                    # print(f'composite score: {score}')
                else:
                    tqdm.write(f"    Skipping unknown metric: {metric_name}")
                    continue

            results[explainer_name][metric_name].extend(score.tolist())


        # --- Batch Clean up ---
        del inputs, labels, outputs, pred_label_idx
        if torch.cuda.is_available():
             torch.cuda.empty_cache()


        # Optional: Break early for debugging
        # if batch_idx >= 2:
        #    print("Stopping early for debugging.")
        #    break

    for explainer_key, metric_data in results.items():
        savedir_evals = os.path.join(savedir_base, f'evals/{explainer_key}.pkl')
        save_pickle_data(results[explainer_key], savedir_evals)


def visualize(args):
    import matplotlib.cm as cm
    from itertools import chain

    TARGET_METRIC_KEYS = ['ab_pc', 'complexity', 'composite']
    cwd = os.getcwd()
    datadir = os.path.join(cwd, f'results/omnixai/{args.model}')

    eval_results = defaultdict(dict)
    bar_means = defaultdict(list)
    bar_stds = defaultdict(list)

    for explainer_key in args.eval_explainer:
        eval_results[explainer_key] = load_pickle_data(os.path.join(datadir, f'evals/{explainer_key}.pkl'))
        
        for target_metric_key in TARGET_METRIC_KEYS:
            bar_means[target_metric_key].append(np.array(eval_results[explainer_key][target_metric_key]).mean())
            bar_stds[target_metric_key].append(np.array(eval_results[explainer_key][target_metric_key]).std())

    # exit()

    for target_metric_key in TARGET_METRIC_KEYS:
        formatted_means = [f'{mean:.3f}' for mean in bar_means[target_metric_key]]
        formatted_stds = [f'{std:.3f}' for std in bar_stds[target_metric_key]]
        print(f'[{target_metric_key}]\nmean: {formatted_means}\nstd: {formatted_stds}\n')

    bar_width = 0.5
    r = np.arange(len(args.eval_explainer))
    num_bars = len(args.eval_explainer)
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i/num_bars) for i in range(num_bars)]

    for target_metric_key in TARGET_METRIC_KEYS:
        plt.figure(figsize=(14, 8))
        plt.bar(r, bar_means[target_metric_key], yerr=bar_stds[target_metric_key], width=bar_width, edgecolor='grey', color=colors, capsize=5)
        plt.xlabel('Explanation Method', fontsize=12)
        metric_display_name = target_metric_key.replace('_', ' ').title()
        plt.ylabel(f'{metric_display_name} Score (Mean ± Std Dev)', fontsize=12)
        
        plt.xticks([r_ for r_ in r], args.eval_explainer, rotation=30, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        savedir_quant = os.path.join(datadir, f'evals/figures/opt_evals_{target_metric_key}.jpg')
        plt.savefig(savedir_quant)
        plt.close()
        print(f'Evaluation bar chart saved to: {savedir_quant}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate XAI methods using OmniXAI")
    parser.add_argument('--model', type=str, choices=TORCHVISION_MODEL_CHOICES, required=True, help="Torchvision model name")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ImageNet validation dataset")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation")
    parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU usage")
    parser.add_argument('--evaluate', action='store_true', help="Run the evaluation process")
    parser.add_argument('--visualize', action='store_true', help="Run the visualization process")
    parser.add_argument('--eval_explainer', type=str, default=None, nargs='+', help=f"Evaluate only specific explainers by name")

    args = parser.parse_args()

    if not args.evaluate and not args.visualize:
        args.evaluate = True
        args.visualize = True

    if args.evaluate:
        evaluate(args)
        print("\nEvaluation finished.")

    if args.visualize:
        visualize(args)
        print("\nVisualization finished.")
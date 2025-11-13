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

# --- AutoXAI Imports ---
from vendor.AutoXAI.launch import main

# --- pnpxai Metric Imports ---
try:
    from pnpxai.evaluator.metrics import AbPC, Complexity, Simplicity, CompositeMetrics, MoRF, LeRF
except ImportError as e:
    print(f"Error importing pnpxai metrics: {e}")
    exit()

# --- Helper Imports ---
try:
    from tutorials.img.helpers import get_torchvision_model, get_imagenet_val_dataset, denormalize_image, save_pickle_data, load_pickle_data
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


# --- Evaluation Function using Captum ---
def evaluate(args):
    # setup
    cwd = os.getcwd()
    savedir_base = os.path.join(cwd, f'results/autoxai/{args.model}')
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

    # --- Define Captum Explainers ---
    # target_layer = _find_target_layer(model, args.model)
    cam_target_layer = find_cam_target_layer(model)

    # NoiseTunnel configuration (used for SmoothGrad, VarGrad)
    smooth_grad_nt = NoiseTunnel(Saliency(model))
    var_grad_nt = NoiseTunnel(Saliency(model)) # VarGrad might need specific noise_type if different

    captum_explainers = {
        'gradient': Saliency(model),
        'gradient_x_input': InputXGradient(model),
        'smooth_grad': smooth_grad_nt, # Requires running attribute with nt_type='smoothgrad'
        'var_grad': var_grad_nt, # Requires running attribute with nt_type='vargrad'

        # LRP Variants - Using basic LRP, rules can be customized
        # 'lrp_epsilon_alpha2_beta1': LRP(model), # Default EpsilonRule, adjust if needed
        # 'lrp_epsilon_gamma_box': LRP(model),    # Needs GammaRule potentially
        # 'lrp_epsilon_plus': LRP(model),         # Needs EpsilonRule(epsilon=?)
        # 'lrp_uniform_epsilon': LRP(model),      # Needs EpsilonRule(epsilon=?)
        'lrp': LRP(model),

        # Layer-based methods
        'grad_cam': LayerGradCam(model, cam_target_layer) if cam_target_layer else None,
        'guided_grad_cam': GuidedGradCam(model, cam_target_layer) if cam_target_layer else None,

        # Perturbation/Sampling-based (Potentially Slow)
        'kernel_shap': KernelShap(model), # Uncomment if needed (SLOW)
        'lime': Lime(model),             # Uncomment if needed (SLOW, requires more setup)
        'integrated_gradients': IntegratedGradients(model), # If needed
        # 'rap': None, # RAP is not implemented in Captum
    }

    # Filter out None explainers
    active_explainers = defaultdict(dict)
    for explainer in args.eval_explainer:
        if explainer in captum_explainers:
             print(f"Evaluating explainer: {explainer}")
             active_explainers[explainer] = captum_explainers[explainer]
        else:
            print(f"Requested explainer '{explainer}' not available or inactive. Evaluating all active ones.")

    print(f"Active explainers for evaluation: {list(active_explainers.keys())}")
    if 'rap' not in captum_explainers:
         print("Note: RAP explainer is not available in Captum and will be skipped.")
    if cam_target_layer is None and ('grad_cam' in captum_explainers or 'guided_grad_cam' in captum_explainers):
        print("Warning: Could not find target layer. GradCAM and GuidedGradCAM will be skipped.")


    # --- Instantiate pnpxai Metrics ---
    print("Instantiating metrics...")
    try:
        # Metrics instantiation remains the same as it uses pnpxai
        abpc_metric = AbPC(model=model, target_input_keys=[0])

        # simplicity_metric = Simplicity(model=model)
        complexity_metric = Complexity(model=model)

        composite_metric = CompositeMetrics(model=model, target_input_keys=[0])
        composite_metric.composite_metrics = [AbPC(model=model, target_input_keys=[0]), Simplicity(model=model)]
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
    print("\n--- Starting Captum Evaluation ---")
    results = defaultdict(lambda: defaultdict(list)) # results[explainer][metric] = [scores]

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating Batches")):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad_() # Ensure inputs require grad for gradient-based methods

        # Get model prediction (needed for target index) - use no_grad here
        with torch.no_grad():
            outputs = model(inputs)
            pred_label_idx = torch.argmax(outputs, dim=1)

        # --- Baselines (if needed by some methods) ---
        # Example: Zero baseline
        baselines = torch.zeros_like(inputs)

        for explainer_name, explainer_instance in active_explainers.items():
            print(explainer_name)
            tqdm.write(f"  Batch {batch_idx+1}/{len(dataloader)}, Explainer: {explainer_name}")

            # --- Compute Attributions ---
            current_attribution = None
            if isinstance(explainer_instance, Saliency) \
            or isinstance(explainer_instance, InputXGradient) \
            or isinstance(explainer_instance, IntegratedGradients): # Basic gradient methods
                current_attribution = explainer_instance.attribute(inputs, target=pred_label_idx)

            elif isinstance(explainer_instance, LRP):
                current_attribution = explainer_instance.attribute(inputs, target=pred_label_idx, verbose=False)

            elif isinstance(explainer_instance, LayerGradCam) \
              or isinstance(explainer_instance, GuidedGradCam): # Layer methods
                current_attribution = explainer_instance.attribute(inputs, target=pred_label_idx)
                current_attribution = LayerAttribution.interpolate(
                    layer_attribution=current_attribution,
                    interpolate_dims=(inputs.shape[2:]),
                )

            elif isinstance(explainer_instance, NoiseTunnel): # SmoothGrad / VarGrad
                nt_type = 'smoothgrad' if explainer_name == 'smooth_grad' else 'vargrad'
                # Define std dev and number of samples
                current_attribution = explainer_instance.attribute(
                    inputs,
                    nt_type=nt_type,
                    # nt_samples=5,
                    # stdevs=1.0,
                    target=pred_label_idx
                )

            elif isinstance(explainer_instance, KernelShap): # SLOW
                current_attribution = explainer_instance.attribute(
                    inputs,
                    target=pred_label_idx,
                    # baselines=baselines,
                    # n_samples=25
                )

            elif isinstance(explainer_instance, Lime): # SLOW
                current_attribution = explainer_instance.attribute(
                    inputs,
                    target=pred_label_idx,
                    # feature_mask=None, # Provide mask if needed
                    # n_samples=25
                )

            if current_attribution is None:
                tqdm.write(f"    Skipping metrics for {explainer_name} (attribution failed or type not handled)")
                continue

            # --- Postprocessing ---
            pp_attribution = normalize_attr(current_attribution.detach().cpu().numpy(), sign='absolute_value', reduction_axis=1)
            pp_attribution = torch.tensor(pp_attribution)

            # fig, axs = plt.subplots(8, 2, figsize=(6, 24))
            # for i in range(8):
            #     axs[i][0].imshow(inputs[i].permute(1, 2, 0).detach().cpu().numpy())
            #     # axs[i][1].imshow(current_attribution[i].abs().mean(dim=0).detach().cpu().numpy(), cmap='Reds')
            #     axs[i][1].imshow(pp_attribution[i], cmap='Reds')
            #     [ax.axis('off') for ax in axs[i]]
            #     fig.suptitle(explainer_name)
            # plt.tight_layout()
            # plt.savefig(f'captum_attrs_debug/captum_{explainer_name}.jpg')
            # plt.close()
            # continue

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
        del inputs, labels, outputs, pred_label_idx, baselines
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
    datadir = os.path.join(cwd, f'results/captum/{args.model}')

    eval_results = defaultdict(dict)
    bar_means = defaultdict(list)
    bar_stds = defaultdict(list)

    for explainer_key in args.eval_explainer:
        eval_results[explainer_key] = load_pickle_data(os.path.join(datadir, f'evals/{explainer_key}.pkl'))

        # for target_metric_key in TARGET_METRIC_KEYS:
        #     eval_results[explainer_key][target_metric_key] = list(chain.from_iterable(eval_results[explainer_key][target_metric_key]))

        # savedir_evals = os.path.join(datadir, f'evals/flattened/{explainer_key}.pkl')
        # save_pickle_data(eval_results[explainer_key], savedir_evals)

        # continue
        
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
    parser = argparse.ArgumentParser(description="Evaluate XAI methods using Captum")
    parser.add_argument('--model', type=str, choices=TORCHVISION_MODEL_CHOICES, required=True, help="Torchvision model name")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ImageNet validation dataset")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation")
    parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU usage")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--eval_explainer', type=str, default=None, nargs='+', help="Evaluate only a specific explainer by name")

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
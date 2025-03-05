'''
CUDA_VISIBLE_DEVICES=3 python auto_explanation_livertumor_hpovis.py \
    --start_idx 900 \
    --data_id 300 \
    --n_trials 100 \
    --visualize
'''
import os
import re
import copy
import argparse
import glob
import pickle
import json
from easydict import EasyDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

import torch
from torch.utils.data import DataLoader

from pnpxai import AutoExplanationForImageClassification
from pnpxai.evaluator.metrics import RelevanceAccuracy
from pnpxai.utils import set_seed, ignore_warnings
from helpers import get_livertumor_dataset, get_livertumor_model, denormalize_sample

import pdb

#------------------------------------------------------------------------------#
#-------------------------------- basic usage ---------------------------------#
#------------------------------------------------------------------------------#

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int)
# parser.add_argument('--data_ids', nargs='+', type=int, help='List of data IDs')
# parser.add_argument('--data_id', default=100, type=int)
# parser.add_argument('--explainer_id', default=7, type=int)
parser.add_argument('--n_trials', default=100, type=int)
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--visualize', action='store_true')

args = parser.parse_args()

if args.start_idx == 3600:
    data_ids = range(0, 386, 20)
    indices = range(args.start_idx, 3986, 1)
else:
    data_ids = range(0, 400, 20)
    indices = range(args.start_idx, args.start_idx+400, 1)


# setup
set_seed(0)
ignore_warnings()
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spec = 'v4/Rot_Res'
model, transform = get_livertumor_model(f'../models/liver_tumor/{spec}/best_model.pt')

if args.save_data:
    dataset = get_livertumor_dataset(
        transform=transform,
        root_dir='./data/LiverTumor/L00_T20_W',
        indices=indices,
    )
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=False)

    # create auto explanation
    expr = AutoExplanationForImageClassification(
        model=model.to(device),
        data=loader,
        input_extractor=lambda batch: batch[1].to(device),
        label_extractor=lambda batch: batch[2].to(device),
        target_extractor=lambda outputs: outputs.argmax(-1).to(device),
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

    expr.manager.clear_metrics([0, 2, 3])


#------------------------------------------------------------------------------#
#------------------------------- optimization ---------------------------------#
#------------------------------------------------------------------------------#

# user inputs

AVAILABLE_KEYS = ['inputs', 'labels', 'outputs', 'targets', 'explanation', 'postprocessed', 'evaluation']

if args.save_data:
    explainer_ids = range(len(copy.deepcopy(expr.manager.explainers)))
    expr.predict_batch(data_ids)

    n_trials = args.n_trials
    metric_id = 0 # metric_id to be used as objective: AbPC
    postprocessor_id = 0 # sumpos

    for data_id in data_ids:
        inputs, w_inputs, labels, masks = expr.manager.batch_data_by_ids(data_ids=[data_id])
        inputs, w_inputs = inputs.to(device), w_inputs.to(device)
        targets = expr.manager.batch_outputs_by_ids(data_ids=[data_id]).argmax(-1).to(device)

        # Save data only if it is labeled as tumor
        if labels.item() != 1:
            continue

        savedir = f'./hpo_experiments/{args.start_idx+data_id}'
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(
            {'inputs': inputs, 'w_inputs': w_inputs, 'labels': labels, 'masks': masks, 'targets': targets},
            open(os.path.join(savedir, f'data.pkl'), 'wb')
        )

        for explainer_id in explainer_ids:
            print(f'\n---- Running experiment with explainer_id: {explainer_id} ----')

            results = expr.run_batch(
                data_ids=[data_id], # data_ids,
                explainer_id=explainer_id,
                postprocessor_id=postprocessor_id,
                metric_id=metric_id,
            )
            pickle.dump(
                {k: v for k, v in results.items() if k in AVAILABLE_KEYS},
                open(os.path.join(savedir, f'results_{explainer_id}.pkl'), 'wb')
            )

            # optimize: returns optimal explainer id, optimal postprocessor id, (and study)
            optimized, objective, study = expr.optimize(
                data_id=data_id,
                explainer_id=explainer_id,
                metric_id=metric_id,
                direction='maximize',
                sampler='tpe',
                n_trials=n_trials,
                seed=0,
            )
            values = [trial.value for trial in study.trials if trial.value is not None]
            trials = np.arange(len(values))
            pickle.dump(
                values, open(os.path.join(savedir, f'hpo_{explainer_id}.pkl'), 'wb')
            )


            # explain and evaluate with optimal explainer and postprocessor
            opt_results = expr.run_batch(
                data_ids=[data_id],
                # explainer_id=explainer_id,
                explainer_id=optimized['explainer_id'],
                postprocessor_id=optimized['postprocessor_id'],
                metric_id=metric_id, # any metric to evaluate the optimized explanation
            )
            pickle.dump(
                {k: v for k, v in opt_results.items() if k in AVAILABLE_KEYS},
                open(os.path.join(savedir, f'opt_results_{explainer_id}.pkl'), 'wb')
            )

#------------------------------------------------------------------------------#
#------------------------------- visualization --------------------------------#
#------------------------------------------------------------------------------#

'''
class HandlerMultiColorLine(HandlerLine2D):
    def __init__(self, colors, **kwargs):
        self.colors = colors
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = plt.Line2D([xdescent, xdescent + width], [ydescent + height / 2, ydescent + height / 2],
                          linestyle=orig_handle.get_linestyle(), linewidth=orig_handle.get_linewidth())
        line.set_transform(trans)
        line.set_clip_box(legend.get_frame().get_clip_box())
        line.set_color(self.colors)
        return [line]
'''
class HandlerMultiColorLine(HandlerLine2D):
    def __init__(self, colors, **kwargs):
        self.colors = colors
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        num_colors = len(self.colors)
        segment_width = width / num_colors
        lines = []
        for i, color in enumerate(self.colors):
            line = plt.Line2D([xdescent + i * segment_width, xdescent + (i + 1) * segment_width], 
                              [ydescent + height / 2, ydescent + height / 2],
                              linestyle=orig_handle.get_linestyle(), linewidth=orig_handle.get_linewidth(), color=color)
            line.set_transform(trans)
            line.set_clip_box(legend.get_frame().get_clip_box())
            lines.append(line)
        return lines


if args.visualize:
    datadir = 'hpo_experiments'
    all_dirs = glob.glob(os.path.join(datadir, '*'))
    all_dirs = [d for d in all_dirs if os.path.isdir(d) and re.match(r'hpo_experiments/\d+$', d)]
    # data_ids = sorted([int(dir_name.split('/')[-1]) for dir_name in all_dirs])
    # data_ids = [960, 1100, 2260, 3380, 3880]
    data_ids = [960]

    savedir = f'./results/liver_tumor/hpo/v4/'
    os.makedirs(savedir, exist_ok=True)

    METHODS = {
        'GradCam': 'GradCAM',
        'Gradient': 'Gradient',
        'GradientXInput': r'Grad.$\times$Input',
        'GuidedGradCam': 'GuidedGradCAM',
        'IntegratedGradients': 'Int. Gradients',
        'KernelShap': 'KernelSHAP',
        'LRPEpsilonAlpha2Beta1': r'LRP-$\epsilon \alpha_2 \beta_1$',
        'LRPEpsilonGammaBox': r'LRP-$\epsilon \gamma \mathcal{B}$',
        'LRPEpsilonPlus': r'LRP-$\epsilon^+$',
        'LRPUniformEpsilon': r'LRP-Uniform$\epsilon$',
        'Lime': 'LIME',
        'RAP': 'RAP',
        'SmoothGrad': 'SmoothGrad',
        'VarGrad': 'VarGrad',
    }
    TARGET = {0: 'Normal', 1: 'Tumor'}
    explainers = [
        'GradCam', 'Gradient', 'GradientXInput', 'GuidedGradCam', \
        'IntegratedGradients', 'KernelShap', 'LRPEpsilonAlpha2Beta1', \
        'LRPEpsilonGammaBox', 'LRPEpsilonPlus', 'LRPUniformEpsilon', \
        'Lime', 'RAP', 'SmoothGrad', 'VarGrad'
    ]
    titles = [
        'GradCAM', 'Gradient', r'Grad.$\times$Input', 'GuidedGradCAM', \
        'Int. Gradients', 'KernelSHAP', r'LRP-$\epsilon \alpha_2 \beta_1$', \
        r'LRP-$\epsilon \gamma \mathcal{B}$', r'LRP-$\epsilon^+$', r'LRP-Uniform$\epsilon$', \
        'LIME', 'RAP', 'SmoothGrad', 'VarGrad'
    ]
    titles_2 = [
        'GradCAM', 'Gradient', r'Grad.$\times$Input', 'Guided\nGradCAM', \
        'Integrated\nGradients', 'KernelSHAP', r'LRP-$\epsilon \alpha_2 \beta_1$', \
        r'LRP-$\epsilon \gamma \mathcal{B}$', r'LRP-$\epsilon^+$', r'LRP-Uniform$\epsilon$', \
        'LIME', 'RAP', 'SmoothGrad', 'VarGrad'
    ]


    for data_id in data_ids:
        values = []
        attributions = []
        opt_attributions = []
        mass_accuracies = []
        opt_mass_accuracies = []
        rank_accuracies = []
        opt_rank_accuracies = []

        fig = plt.figure(figsize=(28, 10.5))
        gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[3, 1.5, 3], figure=fig, hspace=0.10, wspace=0.)

        gs_inputs = gs[0].subgridspec(nrows=1, ncols=2, hspace=0., wspace=0.05, width_ratios=[2, 3])
        gs_samples = gs_inputs[0].subgridspec(nrows=1, ncols=2, hspace=0., wspace=0.05, width_ratios=[1, 1])
        gs_hpos = gs_inputs[1].subgridspec(nrows=1, ncols=3, hspace=0., wspace=0.3, width_ratios=[1, 1, 1])

        gs_legends = gs[1].subgridspec(nrows=1, ncols=2, hspace=0.1, wspace=0.)

        gs_attrs = gs[2].subgridspec(nrows=2, ncols=14, hspace=0.05, wspace=0.05)
        axes_dict = {}

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['pdf.fonttype'] = 42

        # fontsize_title = 18
        # fontsize_label = 15
        # fontsize_legend = 12
        fontsize_title = 25 # 20
        fontsize_label = 18
        fontsize_legend = 17

        files = glob.glob(f'{datadir}/{data_id}/results_*.pkl')
        # assert len(explainers) <= len(files), f'files: {len(files)} / explainers: {len(explainers)}'
        if len(explainers) > len(files):
            plt.close(fig)
            continue
        relevance_accuracy = RelevanceAccuracy(model, None)

        # Data
        data = EasyDict(pickle.load(open(os.path.join(datadir, str(data_id), 'data.pkl'), 'rb')))
        inputs, w_inputs, labels, masks, targets = data.inputs, data.w_inputs, data.labels, data.masks, data.targets

        for explainer_id in range(len(explainers)):
            # Attributions
            results = EasyDict(pickle.load(open(os.path.join(datadir, str(data_id), f'results_{explainer_id}.pkl'), 'rb')))
            attributions.append(results.postprocessed)

            # Optimized Attributions
            opt_results = EasyDict(pickle.load(open(os.path.join(datadir, str(data_id), f'opt_results_{explainer_id}.pkl'), 'rb')))
            opt_attributions.append(opt_results.postprocessed)

            # HPO Results
            values.append(
                pickle.load(open(os.path.join(datadir, str(data_id), f'hpo_{explainer_id}.pkl'), 'rb'))
            )

            # Relevance Accuracy
            mass_accuracy, rank_accuracy = relevance_accuracy.evaluate(
                inputs=inputs,
                targets=targets,
                attributions=results.postprocessed,
                ground_truth=masks[:, 0],
            )
            mass_accuracies.append(mass_accuracy.item())
            rank_accuracies.append(rank_accuracy.item())
            opt_mass_accuracy, opt_rank_accuracy = relevance_accuracy.evaluate(
                inputs=inputs,
                targets=targets,
                attributions=opt_results.postprocessed,
                ground_truth=masks[:, 0],
            )
            opt_mass_accuracies.append(opt_mass_accuracy.item())
            opt_rank_accuracies.append(opt_rank_accuracy.item())

        # Compute how many explanation methods are improved with respect to relevance accuracy after HPO
        mass_differences = [opt_mass_accuracies[i] - mass_accuracies[i] for i in range(len(explainers))]
        rank_differences = [opt_rank_accuracies[i] - rank_accuracies[i] for i in range(len(explainers))]

        mass_results = {'improved': [], 'on_par': [], 'degraded': []}
        rank_results = {'improved': [], 'on_par': [], 'degraded': []}

        for i, diff in enumerate(mass_differences):
            if diff > 0:
                mass_results['improved'].append(i)
            elif diff < 0:
                mass_results['degraded'].append(i)
            else:
                mass_results['on_par'].append(i)

        for i, diff in enumerate(rank_differences):
            if diff > 0:
                rank_results['improved'].append(i)
            elif diff < 0:
                rank_results['degraded'].append(i)
            else:
                rank_results['on_par'].append(i)

        summary = {
            'mass': {
                'num_improved': len(mass_results['improved']),
                'num_on_par': len(mass_results['on_par']),
                'num_degraded': len(mass_results['degraded']),
                'improved_id': mass_results['improved'],
                'on_par_id': mass_results['on_par'],
                'degraded_id': mass_results['degraded']
            },
            'rank': {
                'num_improved': len(rank_results['improved']),
                'num_on_par': len(rank_results['on_par']),
                'num_degraded': len(rank_results['degraded']),
                'improved_id': rank_results['improved'],
                'on_par_id': rank_results['on_par'],
                'degraded_id': rank_results['degraded']
            }
        }

        with open(os.path.join(savedir, f'{data_id}.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Sliced CT image
        ax = fig.add_subplot(gs_samples[0])
        ax.set_title('Sliced CT Image', fontsize=fontsize_title, fontweight='bold')
        ax.imshow(inputs[0].squeeze().detach().cpu().numpy(), cmap='gray')
        # ax.set_title(f'Label: {TARGET[labels.item()]} / Pred: {TARGET[targets.item()]}', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'Label: {TARGET[labels.item()]} / Pred: {TARGET[targets.item()]}', fontsize=fontsize_label)
        ax.set_xticks([])
        ax.set_yticks([])
        axes_dict[gs_samples[0]] = ax

        # Ground truth segmentation mask
        ax = fig.add_subplot(gs_samples[1])
        ax.set_title('Ground Truth Mask', fontsize=fontsize_title, fontweight='bold')
        ax.imshow(masks[0].squeeze(), cmap='gray')
        # ax.set_title('GT Seg. Mask', fontsize=10, fontweight='bold')
        ax.set_xlabel('Tumor Segments in the Liver', fontsize=fontsize_label)
        ax.set_xticks([])
        ax.set_yticks([])
        axes_dict[gs_samples[1]] = ax

        # Add legend for the segmentation mask
        # ax = fig.add_subplot(gs_inputs[2])
        ax = fig.add_subplot(gs_legends[0])
        legend_elements = [
            mpatches.Patch(facecolor='black', edgecolor='black', label='Background'),
            mpatches.Patch(facecolor='gray', edgecolor='black', label='Healthy Liver'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='Tumor')
        ]
        # ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.58, 0.6), frameon=True, ncols=len(legend_elements))
        # ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.42, 0.6), frameon=True, ncols=len(legend_elements))
        ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.32, 0.5), frameon=True, ncols=2)
        ax.axis('off')
        # axes_dict[gs_inputs[2]] = ax
        axes_dict[gs_legends[0]] = ax

        # HPO Results
        ax = fig.add_subplot(gs_hpos[0])
        ax.set_title('Hyperparameter Optimization', fontsize=fontsize_title, fontweight='bold')

        trials = np.arange(len(values[0]))
        # colors = matplotlib.colormaps.get_cmap('tab20', len(explainers))
        colors = plt.cm.get_cmap('tab20', len(explainers))
        for explainer_id in range(len(explainers)):
            ax.scatter(trials, values[explainer_id], alpha=0.1, color=[colors(explainer_id)])
            best_value = np.max(values[explainer_id])
            ax.axhline(y=best_value, color=colors(explainer_id), linestyle='-', linewidth=2)
        
        dummy_line = plt.Line2D([0], [0], color='black', linestyle='-', linewidth=4, label='Best Value for Each Explainer')
        ax.add_line(dummy_line)
        ax.set_xlabel('Trials', fontsize=fontsize_label, fontweight='bold')
        ax.set_ylabel('ABPC', fontsize=fontsize_label, fontweight='bold')

        y_min = min([min(value) for value in values])
        y_max = max([max(value) for value in values])
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

        ax.grid(True)
        # ax.legend(loc='best', fontsize=fontsize_legend)

        ax.legend(
            [dummy_line],
            ['Best Value for Each Explainer'],
            handler_map={dummy_line: HandlerMultiColorLine([colors(i) for i in range(len(explainers))])},
            loc='best',
            fontsize=fontsize_legend
        )

        axes_dict[gs_hpos[0]] = ax

        # Relevance Mass Accuracy
        fig.text(0.75, 0.89, 'Ground Truth Evaluation', ha='center', fontsize=fontsize_title, fontweight='bold')
        ax = fig.add_subplot(gs_hpos[1])

        bar_width = 0.35

        r1 = np.arange(len(explainers))
        r2 = [x + bar_width for x in r1]

        ax.bar(
            r1, mass_accuracies, color=[colors(i) for i in range(len(explainers))],
            width=bar_width, edgecolor='grey', alpha=0.3, # label='Before Optimization',
        )
        ax.bar(
            r2, opt_mass_accuracies, color=[colors(i) for i in range(len(explainers))],
            width=bar_width, edgecolor='grey', alpha=1., # label='After Optimization',
        )
        # ax.bar(labels, mass_accuracies)
        # ax.set_xlabel('Optimization Stage')
        ax.set_xlabel('Attribution Methods', fontsize=fontsize_label, fontweight='bold')
        ax.set_ylabel('Relevance Mass Accuracy', fontsize=fontsize_label, fontweight='bold')
        # ax.set_ylim(-0.05, 1.05)
        ax.set_ylim(-0.02, 0.92)
        ax.set_xticks([r + bar_width/2 for r in range(len(explainers))])
        ax.set_xticklabels([])
        # ax.set_xticklabels(titles, rotation=45, ha='right')
        
        # ax.legend()
        ax.grid(True)
        axes_dict[gs_hpos[1]] = ax

        # Relevance Rank Accuracy
        ax = fig.add_subplot(gs_hpos[2])

        bar_width = 0.35

        r1 = np.arange(len(explainers))
        r2 = [x + bar_width for x in r1]

        colors = plt.cm.get_cmap('tab20', len(explainers))
        ax.bar(
            r1, rank_accuracies, color=[colors(i) for i in range(len(explainers))],
            width=bar_width, edgecolor='grey', alpha=0.3, # label='Before Optimization',
        )
        ax.bar(
            r2, opt_rank_accuracies, color=[colors(i) for i in range(len(explainers))],
            width=bar_width, edgecolor='grey', alpha=1., # label='After Optimization',
        )
        # ax.bar(labels, mass_accuracies)
        # ax.set_xlabel('Optimization Stage')
        ax.set_xlabel('Attribution Methods', fontsize=fontsize_label, fontweight='bold')
        ax.set_ylabel('Relevance Rank Accuracy', fontsize=fontsize_label, fontweight='bold')
        # ax.set_ylim(-0.05, 1.05)
        ax.set_ylim(-0.02, 0.92)
        ax.set_xticks([r + bar_width/2 for r in range(len(explainers))])
        ax.set_xticklabels([])
        # ax.set_xticklabels(titles, rotation=45, ha='right')
        
        # ax.legend()
        ax.grid(True)
        axes_dict[gs_hpos[2]] = ax

        # Add legend for the explanation methods
        ax = fig.add_subplot(gs_legends[1])
        legend_elements = [mpatches.Patch(facecolor=colors(i), edgecolor='grey', label=f'{titles[i]}') for i in range(len(explainers))]
        # ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.4, 0.4), frameon=True, ncols=len(legend_elements)//2)
        # ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.35, 0.4), frameon=True, ncols=len(legend_elements)//2)
        ax.legend(handles=legend_elements, loc='center', fontsize=fontsize_legend, bbox_to_anchor=(0.25, 0.5), frameon=True, ncols=len(legend_elements)//2)
        ax.axis('off')
        axes_dict[gs_legends[1]] = ax

        for explainer_id in range(len(explainers)):
            ax = fig.add_subplot(gs_attrs[0, explainer_id])
            ax.imshow(attributions[explainer_id].squeeze().detach().cpu().numpy(), cmap='Reds')
            if explainer_id == 0:
                ax.set_ylabel('Default', fontsize=fontsize_label, fontweight='bold')
            # ax.set_title(titles_2[explainer_id], fontsize=fontsize_legend, fontweight='bold')
            ax.set_title(titles_2[explainer_id], fontsize=fontsize_label, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(gs_attrs[1, explainer_id])
            ax.imshow(opt_attributions[explainer_id].squeeze().detach().cpu().numpy(), cmap='Reds')
            if explainer_id == 0:
                ax.set_ylabel('Optimized', fontsize=fontsize_label, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])


        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(savedir, f'{data_id}.pdf'), bbox_inches='tight', dpi=300)
        plt.close(fig)
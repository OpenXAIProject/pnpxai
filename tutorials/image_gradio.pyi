import gradio as gr
from gradio.components.base import Component as GrComponent
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio.components import Timer
from gradio.events import Events
from pnpxai.core.experiment import AutoExplanation
from pnpxai.core.detector import extract_graph_data, symbolic_trace
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx


PLOT_PER_LINE = 4
N_FEATURES_TO_SHOW = 5
OPT_N_TRIALS = 10

class App:
    def __init__(self):
        pass

class Component:
    def __init__(self):
        pass

class Tab(Component):
    def __init__(self):
        pass

class OverviewTab(Tab):
    def __init__(self):
        pass

    def show(self):
        with gr.Tab(label="Overview") as tab:
            gr.Label("This is the overview tab.")

class DetectionTab(Tab):
    def __init__(self, experiments):
        self.experiments = experiments

    def show(self):
        with gr.Tab(label="Detection") as tab:
            gr.Label("This is the detection tab.")

            for exp in self.experiments:
                detector_res = DetectorRes(exp)
                # detector_res.show()

class LocalExpTab(Tab):
    def __init__(self, experiments):
        self.experiments = experiments

        self.experiment_components = []
        for exp in self.experiments:
            self.experiment_components.append(Experiment(exp))

    def description(self):
        return "This tab shows the local explanation."

    def show(self):
        with gr.Tab(label="Local Explanation") as tab:
            gr.Label("This is the local explanation tab.")

            for i, exp in enumerate(self.experiments):
                self.experiment_components[i].show()

class DetectorRes(Component):
    def __init__(self, experiment):
        graph_module = symbolic_trace(experiment.model)
        self.graph_data = extract_graph_data(graph_module)

    def describe(self):
        return "This component shows the detection result."

    def show(self):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node in self.graph_data['nodes']:
            G.add_node(node['name'], label=node['target'].split("(")[0])

        # Add edges
        for edge in self.graph_data['edges']:
            G.add_edge(edge['source'], edge['target'])

        # Draw the graph
        pos = nx.spring_layout(G)  # positions for all nodes

        edge_trace = []
        for edge in G.edges():
            edge_trace.append(
                go.Scatter(
                    x=[pos[edge[0]][0], pos[edge[1]][0], None],
                    y=[pos[edge[0]][1], pos[edge[1]][1], None],
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'))

        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[G.nodes[node]['label'] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
            )
        )

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            title='Directed Acyclical Graph Visualization',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )
        graph = gr.Plot(value=fig, label="Model Graph")

class ImgGallery(Component):
    def __init__(self, imgs):
        self.imgs = imgs
        self.selected_index = gr.Number(label="Selected Index", value=0, visible=False)
    
    def on_select(self, evt: gr.SelectData):
        return evt.index

    def show(self):
        gallery = gr.Gallery(value=self.imgs, label="Images", columns=6, height=200)
        gallery.select(self.on_select, outputs=self.selected_index)


class Experiment(Component):
    def __init__(self, experiment):
        self.experiment = experiment

    def viz_input(self, input, data_id):
        orig_img_np = self.experiment.input_visualizer(input)
        orig_img = px.imshow(orig_img_np)

        orig_img.update_layout(
            title=f"Data ID: {data_id}",
            width=400,
            height=350,
            xaxis=dict(
                showticklabels=False,
                ticks='',
                showgrid=False
            ),
            yaxis=dict(
                showticklabels=False,
                ticks='',
                showgrid=False
            ),
        )

        return orig_img


    def get_prediction(self, record, topk=3):
        probs = record['output'].softmax(-1).detach().numpy()
        text = f"Ground Truth Label: {self.experiment.target_visualizer(record['label'])}\n"

        for ind, pred in enumerate(probs.argsort()[-topk:][::-1]):
            label = self.experiment.target_visualizer(torch.tensor(pred))
            prob = probs[pred]
            text += f"Top {ind+1} Prediction: {label} ({prob:.2f})\n"

        return text
    
    def get_exp_plot(self, data_index, exp_res):
        return ExpRes(data_index, exp_res).show()
    
    def rank_explainers(self, record):
        all_metrics, all_metric_ids = self.experiment.manager.get_metrics()

    def generate_record(self, data_id):
        new_record = {}
        records = self.experiment.records

        new_record['data_id'] = records[-1]['data_id']
        new_record['input'] = records[-1]['input']
        new_record['label'] = records[-1]['label']
        new_record['output'] = records[-1]['output']
        new_record['target'] = records[-1]['target']

        new_record['explanations'] = []
        for record in records:
            if record['data_id'] != data_id: continue
            if record['postprocessor_id'] != 0: continue
            explanation = {
                'explainer_nm': record['explainer'].__class__.__name__,
                'value': record['explanation'],
                'evaluations': []
            }
            for record_eval in record['evaluations']:
                # if record_eval['metric'].__class__.__name__ in ["LeRF", "MoRF"]: continue
                explanation['evaluations'].append({
                    'metric_nm': record_eval['metric'].__class__.__name__,
                    'value' : record_eval['evaluation']
                })
            
            new_record['explanations'].append(explanation)

        # Sort new_record['explanations'] with respect to the metric values
        new_record['explanations'] = sorted(new_record['explanations'], key=lambda x: x['evaluations'][0]['value'], reverse=True)

        return new_record

    def handle_click(self, data_id, explainer_names, metric_names):
        all_explainers, all_explainer_ids = self.experiment.manager.get_explainers()
        all_metrics, all_metric_ids = self.experiment.manager.get_metrics()
        all_explainer_names = [exp.__class__.__name__ for exp in all_explainers]
        all_metric_names = [metric.__class__.__name__ for metric in all_metrics]

        explainer_ids = [all_explainer_ids[all_explainer_names.index(name)] for name in explainer_names]
        metric_ids = [all_metric_ids[all_metric_names.index(name)] for name in metric_names]

        pprs_ids = [0]
        self.experiment.run([data_id], explainer_ids, pprs_ids, metric_ids)

        record = self.generate_record(data_id)
        # orig_img = self.viz_input(record['input'], data_id)

        pred = self.get_prediction(record)
        plots = []
        for exp_res in record['explanations']:
            plots.append(self.get_exp_plot(data_id, exp_res))

        n_rows = len(all_explainers) // PLOT_PER_LINE
        n_rows = n_rows + 1 if len(all_explainers) % PLOT_PER_LINE != 0 else n_rows
        total_plots = n_rows * PLOT_PER_LINE


        if len(record['explanations']) < total_plots:
            for _ in range(total_plots - len(record['explanations'])):
                plots.append(None)

        # return [orig_img, pred] + plots
        return [pred] + plots

    def show(self):
        gr.Label(f"Experiment ({self.experiment.model.__class__.__name__})")

        explainers, _ = self.experiment.manager.get_explainers()
        explainer_names = [exp.__class__.__name__ for exp in explainers]
        # explainer_input = gr.CheckboxGroup(label="Explainers", choices=explainer_names, value=explainer_names)

        explainer_checkbox_group = ExplainerCheckboxGroup(
            choices=explainer_names, 
            experiment=self.experiment
        )
        # explainer_checkbox_group.show()
        
        metrics, _ = self.experiment.manager.get_metrics()
        metrics_names = [metric.__class__.__name__ for metric in metrics]
        metric_input = gr.CheckboxGroup(label="Evaluators", choices=metrics_names, value=metrics_names)

        dset = self.experiment.manager._data.dataset
        imgs = []
        for i in range(len(dset)):
            img = self.experiment.input_visualizer(dset[i][0])
            imgs.append(img)
        gallery = ImgGallery(imgs)
        gallery.show()

        bttn = gr.Button("Explain")

        with gr.Row():
            # orig_img = gr.Plot(label="Original Image")
            # prediction_result = gr.Label(label="Prediction result")
            prediction_result = gr.Textbox(label="Prediction result")
        
        n_rows = len(explainers) // PLOT_PER_LINE
        n_rows = n_rows + 1 if len(explainers) % PLOT_PER_LINE != 0 else n_rows
        plots = []
        for i in range(n_rows):
            with gr.Row():
                for j in range(PLOT_PER_LINE):
                    plots.append(gr.Plot(label=f"Explanation {1+i*PLOT_PER_LINE+j}"))
        
        # bttn.click(
        #     self.handle_click, 
        #     inputs=[
        #         gallery.selected_index,
        #         explainer_checkbox_group,
        #         metric_input,
        #     ], 
        #     # outputs=[orig_img, prediction_result] + plots
        #     outputs=[prediction_result] + plots
        # )
                    
class ExplainerCheckboxGroup(gr.CheckboxGroup):
    def __init__(
        self,
        choices: list[str | int | float | tuple[str, str | int | float]] | None = None,
        experiment = None,
        *,
        value: list[str | float | int] | str | float | int | Callable | None = None,
        type: Literal["value", "index"] = "value",
        label: str | None = None,
        info: str | None = None,
        every: Timer | float | None = None,
        inputs: Component | list[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,

    ):
        super().__init__(
            choices=choices,
            value=value,
            label=label,
            info=info,
            every=every,
            inputs=inputs,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
        )
        self.experiment = experiment

                  
# class ExplainerCheckboxGroup(Component):
#     def __init__(self, explainer_names, experiment):
#         super().__init__()
#         self.explainer_names = explainer_names
#         self.explainer_objs = []
#         self.experiment = experiment
#         self.selected_indices = []
    
#     def show(self):
#         cnt = 0
#         while cnt * PLOT_PER_LINE < len(self.explainer_names):
#             with gr.Row():
#                 for exp_nm in self.explainer_names[cnt*PLOT_PER_LINE:(cnt+1)*PLOT_PER_LINE]:
#                     explainer_obj = ExplainerCheckbox(exp_nm)
#                     self.explainer_objs.append(explainer_obj)
#                     explainer_obj.show()
#                 cnt += 1

#     @property
#     def value(self):
#         return self.selected_indices
    
class ExplainerCheckbox(Component):
    def __init__(self, explainer_name):
        self.explainer_name = explainer_name
        self.optimal_info = {}

    def optimize(self, data_id, experiment):
        explainer_id = experiment.manager.get_explainer_by_name(self.explainer_name)
        metric_id = experiment.manager.get_metric_by_name("AbPC")

        opt_explainer_id, opt_postprocessor_id = experiment.optimize(
            data_id=data_id,
            explainer_id=explainer_id,
            metric_id=metric_id,
            direction='maximize',
            sampler='tpe',
            n_trials=OPT_N_TRIALS,
            return_study=False,
        )

        self.optimal_info['opt_explainer_id'] = opt_explainer_id
        self.optimal_info['opt_postprocessor_id'] = opt_postprocessor_id

        checkbox = gr.Checkbox(label="Optimized Parameter (Optimal)", interactive=True)
        bttn = gr.update(value="Optimized", variant="secondary")

        return [checkbox, bttn]

    def show(self):
        with gr.Accordion(self.explainer_name, open=False):
            default_check = gr.Checkbox(label="Default Parameter", value=True, interactive=True)
            opt_check = gr.Checkbox(label="Optimized Parameter (Not Optimal)")
            bttn = gr.Button(value="Optimize", size="sm", variant="primary")
            bttn.click(self.optimize, outputs=[opt_check, bttn])
        

class ExpRes(Component):
    def __init__(self, data_index, exp_res):
        self.data_index = data_index
        self.exp_res = exp_res

    def show(self):
        value = self.exp_res['value']
        explainer_nm = self.exp_res['explainer_nm']

        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(value.detach().numpy()),
            colorscale='Viridis',
            showscale=False  # remove color bar
        ))

        evaluations = self.exp_res['evaluations']
        metric_values = [f"{eval['metric_nm']}: {eval['value']:.2f}" for eval in evaluations if eval['value'] is not None]
        n = 3
        cnt = 0
        while cnt * n < len(metric_values):
            metric_text = ', '.join(metric_values[cnt*n:cnt*n+n])
            fig.add_annotation(
                x=-0.1,
                y=-0.15 - 0.1 * cnt,
                xref='paper',
                yref='paper',
                text=metric_text,
                showarrow=False,
                font=dict(
                    size=12,
                ),
            )
            cnt += 1


        fig = fig.update_layout(
            title=f"{explainer_nm}",
            width=400,
            height=400,
            xaxis=dict(
                showticklabels=False,
                ticks='',
                showgrid=False
            ),
            yaxis=dict(
                showticklabels=False,
                ticks='',
                showgrid=False
            ),
            margin=dict(t=10, b=10, l=10, r=10),
        )

        return fig


class ImageClsApp(App):
    def __init__(self, experiments, **kwargs):
        self.name = "Image Classification App"
        super().__init__(**kwargs)

        self.experiments = experiments

        self.overview_tab = OverviewTab()
        self.detection_tab = DetectionTab(self.experiments)
        self.local_exp_tab = LocalExpTab(self.experiments)

    def launch(self, **kwargs):
        with gr.Blocks(
            title=self.name,
        ) as demo:
            self.overview_tab.show()
            self.detection_tab.show()
            self.local_exp_tab.show()

        return demo

# if __name__ == '__main__':
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

os.environ['GRADIO_TEMP_DIR'] = '.tmp'

def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())

experiments = []

model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
experiment1 = AutoExplanation(
    model=model,
    data=loader,
    modality='image',
    question='why',
    evaluator_enabled=True,
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    input_visualizer=lambda x: denormalize_image(x, transform.mean, transform.std),
    target_visualizer=target_visualizer,
    target_labels=False, # target prediction if False
    channel_dim=1
)


model, transform = get_torchvision_model('vit_b_16')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
experiment2 = AutoExplanation(
    model=model,
    data=loader,
    modality='image',
    question='why',
    evaluator_enabled=True,
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    input_visualizer=lambda x: denormalize_image(x, transform.mean, transform.std),
    target_visualizer=target_visualizer,
    target_labels=False, # target prediction if False
    channel_dim=1
)


experiments.append(experiment1)
experiments.append(experiment2)

app = ImageClsApp(experiments)
demo = app.launch()
demo.launch()
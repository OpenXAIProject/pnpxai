# python image_gradio.py >> ./logs/image_gradio.log 2>&1
import gradio as gr
from pnpxai.core.experiment import AutoExplanation
from pnpxai.core.detector import extract_graph_data, symbolic_trace
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx


PLOT_PER_LINE = 4
N_FEATURES_TO_SHOW = 5
OPT_N_TRIALS = 10
OBJECTIVE_METRIC = "AbPC"
SAMPLE_METHOD = "tpe"

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
        self.selected_index = gr.Number(value=0, label="Selected Index", visible=False)
    
    def on_select(self, evt: gr.SelectData):
        return evt.index

    def show(self):
        self.gallery_obj = gr.Gallery(value=self.imgs, label="Input Data Gallery", columns=6, height=200)
        self.gallery_obj.select(self.on_select, outputs=self.selected_index)


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
        probs = record['output'].softmax(-1).squeeze().detach().numpy()
        text = f"Ground Truth Label: {self.experiment.target_visualizer(record['label'])}\n"

        for ind, pred in enumerate(probs.argsort()[-topk:][::-1]):
            label = self.experiment.target_visualizer(torch.tensor(pred))
            prob = probs[pred]
            text += f"Top {ind+1} Prediction: {label} ({prob:.2f})\n"
        
        return text


    def get_exp_plot(self, data_index, exp_res):
        return ExpRes(data_index, exp_res).show()
    
    def get_metric_id_by_name(self, metric_name):
        metric_info = self.experiment.manager.get_metrics()
        idx = [metric.__class__.__name__ for metric in metric_info[0]].index(metric_name)
        return metric_info[1][idx]

    def generate_record(self, data_id, metric_names):
        record = {}
        _base = self.experiment.run_batch([data_id], 0, 0, 0)
        record['data_id'] = data_id
        record['input'] = _base['inputs']
        record['label'] = _base['labels']
        record['output'] = _base['outputs']
        record['target'] = _base['targets']
        record['explanations'] = []

        metrics_ids = [self.get_metric_id_by_name(metric_nm) for metric_nm in metric_names]

        cnt = 0
        for info in self.explainer_checkbox_group.info:
            if info['checked']:
                base = self.experiment.run_batch([data_id], info['id'], info['pp_id'], 0)
                record['explanations'].append({
                    'explainer_nm': base['explainer'].__class__.__name__,
                    'value': base['postprocessed'],
                    'mode' : info['mode'],
                    'evaluations': []
                })
                for metric_id in metrics_ids:
                    res = self.experiment.run_batch([data_id], info['id'], info['pp_id'], metric_id)
                    record['explanations'][-1]['evaluations'].append({
                        'metric_nm': res['metric'].__class__.__name__,
                        'value' : res['evaluation']
                    })

                cnt += 1

        # Sort record['explanations'] with respect to the metric values
        record['explanations'] = sorted(record['explanations'], key=lambda x: x['evaluations'][0]['value'], reverse=True)

        return record

    def gen_handle_click(self, data_id, metric_inputs, bttn):
        @gr.render(inputs=[data_id] + metric_inputs, triggers=[bttn.click])
        def inner_func(data_id, *metric_inputs):
            metric_input = []
            for metric in metric_inputs:
                if metric:
                    metric_input += metric
                    
            record = self.generate_record(data_id, metric_input)

            pred = self.get_prediction(record)
            gr.Textbox(label="Prediction result", value=pred)
            

            n_rows = len(record['explanations']) // PLOT_PER_LINE
            n_rows = n_rows + 1 if len(record['explanations']) % PLOT_PER_LINE != 0 else n_rows
            plots = []
            figs = []
            for i in range(n_rows):
                with gr.Row():
                    for j in range(PLOT_PER_LINE):
                        if i*PLOT_PER_LINE+j < len(record['explanations']):
                            exp_res = record['explanations'][i*PLOT_PER_LINE+j]
                            fig = self.get_exp_plot(data_id, exp_res)
                            plot_obj = gr.Plot(value=go.Figure(), label=f"{exp_res['explainer_nm']} ({exp_res['mode']})", visible=False)
                            plots.append(plot_obj)
                            figs.append(fig)
                        else:
                            plots.append(gr.Plot(value=None, label="Blank", visible=False))


            def show_result():
                _plots = []
                for i in range(n_rows):
                    for j in range(PLOT_PER_LINE):
                        if i*PLOT_PER_LINE+j < len(record['explanations']):
                            _plots.append(gr.Plot(value=figs[i*PLOT_PER_LINE+j], visible=True))
                        else:
                            _plots.append(gr.Plot(value=None, visible=True))
                return _plots
            
            invisible = gr.Number(value=0, visible=False)
            invisible.change(show_result, outputs=plots)
            invisible.value = 1

        return inner_func


    def show(self):
        with gr.Row():
            gr.Textbox(value="Image Classficiation", label="Task")
            gr.Textbox(value=f"{self.experiment.model.__class__.__name__}", label="Model")
            gr.Textbox(value="Heatmap", label="Explanation Type")

        dset = self.experiment.manager._data.dataset
        imgs = []
        for i in range(len(dset)):
            img = self.experiment.input_visualizer(dset[i][0])
            imgs.append(img)
        gallery = ImgGallery(imgs)
        gallery.show()

        explainers, _ = self.experiment.manager.get_explainers()
        explainer_names = [exp.__class__.__name__ for exp in explainers]

        self.explainer_checkbox_group = ExplainerCheckboxGroup(explainer_names, self.experiment, gallery)
        self.explainer_checkbox_group.show()
        
        cr_metrics_names = ["AbPC", "MoRF", "LeRF", "MuFidelity"]
        cn_metrics_names = ["Sensitivity"]
        cp_metrics_names = ["Complexity"]
        with gr.Accordion("Evaluators", open=True):
            with gr.Row():
                cr_metrics = gr.CheckboxGroup(choices=cr_metrics_names, value=cr_metrics_names, label="Correctness")
            with gr.Row():
                # cn_metrics = gr.CheckboxGroup(choices=cn_metrics_names, value=cn_metrics_names, label="Continuity")
                cn_metrics = gr.CheckboxGroup(choices=cn_metrics_names, label="Continuity")
            with gr.Row():
                cp_metrics = gr.CheckboxGroup(choices=cp_metrics_names, value=cp_metrics_names, label="Compactness")

        metric_inputs = [cr_metrics, cn_metrics, cp_metrics]

        bttn = gr.Button("Explain", variant="primary")
        handle_click = self.gen_handle_click(gallery.selected_index, metric_inputs, bttn)

                    
class ExplainerCheckboxGroup(Component):
    def __init__(self, explainer_names, experiment, gallery):
        super().__init__()
        self.explainer_names = explainer_names
        self.explainer_objs = []
        self.experiment = experiment
        self.gallery = gallery
        explainers, exp_ids = self.experiment.manager.get_explainers()

        self.info = []
        for exp, exp_id in zip(explainers, exp_ids):
            self.info.append({'nm': exp.__class__.__name__, 'id': exp_id, 'pp_id' : 0, 'mode': 'default', 'checked': True})

    def update_check(self, exp_id, val=None):
        for info in self.info:
            if info['id'] == exp_id:
                if val is not None:
                    info['checked'] = val
                else:
                    info['checked'] = not info['checked']

    def insert_check(self, exp_nm, exp_id, pp_id):
        if exp_id in [info['id'] for info in self.info]:
            return

        self.info.append({'nm': exp_nm, 'id': exp_id, 'pp_id' : pp_id, 'mode': 'optimal', 'checked': False})

    def update_gallery_change(self):
        checkboxes = []
        checkboxes += [gr.Checkbox(label="Default Parameter", value=True, interactive=True)] * len(self.explainer_objs)
        checkboxes += [gr.Checkbox(label="Optimized Parameter (Not Optimal)", interactive=False)] * len(self.explainer_objs)
        return checkboxes

    def get_checkboxes(self):
        checkboxes = []
        checkboxes += [exp.default_check for exp in self.explainer_objs]
        checkboxes += [exp.opt_check for exp in self.explainer_objs]
        return checkboxes
    
    def show(self):
        cnt = 0
        with gr.Accordion("Explainers", open=True):
            while cnt * PLOT_PER_LINE < len(self.explainer_names):
                with gr.Row():
                    for info in self.info[cnt*PLOT_PER_LINE:(cnt+1)*PLOT_PER_LINE]:
                        explainer_obj = ExplainerCheckbox(info['nm'], self, self.experiment, self.gallery)
                        self.explainer_objs.append(explainer_obj)
                        explainer_obj.show()
                    cnt += 1
        
        checkboxes = self.get_checkboxes()
        self.gallery.gallery_obj.select(
            fn=self.update_gallery_change,
            outputs=checkboxes
        )

    
class ExplainerCheckbox(Component):
    def __init__(self, explainer_name, groups, experiment, gallery):
        self.explainer_name = explainer_name
        self.groups = groups
        self.experiment = experiment
        self.gallery = gallery
        
        self.default_exp_id = self.get_explainer_id_by_name(explainer_name)
        self.obj_metric = self.get_metric_id_by_name(OBJECTIVE_METRIC)

    def get_explainer_id_by_name(self, explainer_name):
        explainer_info = self.experiment.manager.get_explainers()
        idx = [exp.__class__.__name__ for exp in explainer_info[0]].index(explainer_name)
        return explainer_info[1][idx]
    
    def get_metric_id_by_name(self, metric_name):
        metric_info = self.experiment.manager.get_metrics()
        idx = [metric.__class__.__name__ for metric in metric_info[0]].index(metric_name)
        return metric_info[1][idx]


    def optimize(self):
        data_id = self.gallery.selected_index

        opt_explainer_id, opt_postprocessor_id = self.experiment.optimize(
            data_id=data_id.value,
            explainer_id=self.default_exp_id,
            metric_id=self.obj_metric,
            direction='maximize',
            sampler=SAMPLE_METHOD,
            n_trials=OPT_N_TRIALS,
            return_study=False,
        )

        self.groups.insert_check(self.explainer_name, opt_explainer_id, opt_postprocessor_id)
        self.optimal_exp_id = opt_explainer_id
        checkbox = gr.update(label="Optimized Parameter (Optimal)", interactive=True)
        bttn = gr.update(value="Optimized", variant="secondary")

        return [checkbox, bttn]

    def default_on_select(self, evt: gr.EventData):
        self.groups.update_check(self.default_exp_id, evt._data['value'])

    def optimal_on_select(self, evt: gr.EventData):
        if hasattr(self, "optimal_exp_id"):
            self.groups.update_check(self.optimal_exp_id, evt._data['value'])
        else:
            raise ValueError("Optimal explainer id is not found.")

    def show(self):
        with gr.Accordion(self.explainer_name, open=False):
            self.default_check = gr.Checkbox(label="Default Parameter", value=True, interactive=True)
            self.opt_check = gr.Checkbox(label="Optimized Parameter (Not Optimal)", interactive=False)

            self.default_check.select(self.default_on_select)
            self.opt_check.select(self.optimal_on_select)

            bttn = gr.Button(value="Optimize", size="sm", variant="primary")
            bttn.click(self.optimize, outputs=[self.opt_check, bttn])
        

class ExpRes(Component):
    def __init__(self, data_index, exp_res):
        self.data_index = data_index
        self.exp_res = exp_res

    def show(self):
        value = self.exp_res['value']

        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(value[0].detach().numpy()),
            colorscale='Reds',
            showscale=False  # remove color bar
        ))

        evaluations = self.exp_res['evaluations']
        metric_values = [f"{eval['metric_nm'][:4]}: {eval['value'].item():.2f}" for eval in evaluations if eval['value'] is not None]
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
            title="",
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

    def title(self):
        return """
        <div style="text-align: center;">
            <img src="/file=data/static/XAI-Top-PnP.svg" width="100" height="100">
            <h1> Plug and Play XAI Platform for Image Classification </h1>
        </div>
        """

    def launch(self, **kwargs):
        with gr.Blocks(
            title=self.name,
        ) as demo:
            gr.set_static_paths("./")
            gr.HTML(self.title())

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
demo.launch(favicon_path="data/static/XAI-Top-PnP.svg", share=True)
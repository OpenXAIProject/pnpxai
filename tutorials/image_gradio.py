# TODO:
# 3. Huggingface loading -> after module import system checking
# python image_gradio.py >> ./logs/image_gradio.log 2>&1
# tmux attach -t gd && python image_gradio.py >> ./logs/image_gradio.log 2>&1 && tmux detach
import time
import os
import gradio as gr
from pnpxai.core.experiment import AutoExplanationForImageClassification
from pnpxai.core.detector import extract_graph_data, symbolic_trace
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import secrets


PLOT_PER_LINE = 4
N_FEATURES_TO_SHOW = 5
OPT_N_TRIALS = 10
OBJECTIVE_METRIC = "AbPC"
SAMPLE_METHOD = "tpe"
DEFAULT_EXPLAINER = ["GradientXInput", "IntegratedGradients", "LRPEpsilonPlus"]

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

            for nm, exp_info in self.experiments.items():
                exp = exp_info['experiment']
                detector_res = DetectorRes(exp)
                detector_res.show()

class LocalExpTab(Tab):
    def __init__(self, experiments):
        self.experiments = experiments

        self.experiment_components = []
        for nm, exp_info in self.experiments.items():
            self.experiment_components.append(Experiment(exp_info))

    def description(self):
        return "This tab shows the local explanation."

    def show(self):
        with gr.Tab(label="Local Explanation") as tab:
            gr.Label("This is the local explanation tab.")

            for i, exp in enumerate(self.experiments):
                self.experiment_components[i].show()

class DetectorRes(Component):
    def __init__(self, experiment):
        self.experiment = experiment
        graph_module = symbolic_trace(experiment.model)
        self.graph_data = extract_graph_data(graph_module)

    def describe(self):
        return "This component shows the detection result."
    
    def show(self):
        G = nx.DiGraph()
        root = None
        for node in self.graph_data['nodes']:
            if node['op'] == 'placeholder':
                root = node['name']

            G.add_node(node['name'])


        for edge in self.graph_data['edges']:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'])


        def get_pos1(graph):
            graph = graph.copy()
            for layer, nodes in enumerate(reversed(tuple(nx.topological_generations(graph)))):
                for node in nodes:
                    graph.nodes[node]["layer"] = layer

            pos = nx.multipartite_layout(graph, subset_key="layer", align='horizontal')
            return pos


        def get_pos2(graph, root, levels=None, width=1., height=1.):
            '''
            G: the graph
            root: the root node
            levels: a dictionary
                    key: level number (starting from 0)
                    value: number of nodes in this level
            width: horizontal space allocated for drawing
            height: vertical space allocated for drawing
            '''
            TOTAL = "total"
            CURRENT = "current"

            def make_levels(levels, node=root, currentLevel=0, parent=None):
                # Compute the number of nodes for each level
                if not currentLevel in levels:
                    levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
                levels[currentLevel][TOTAL] += 1
                neighbors = graph.neighbors(node)
                for neighbor in neighbors:
                    if not neighbor == parent:
                        levels = make_levels(levels, neighbor, currentLevel + 1, node)
                return levels

            def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
                dx = 1/levels[currentLevel][TOTAL]
                left = dx/2
                pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
                levels[currentLevel][CURRENT] += 1
                neighbors = graph.neighbors(node)
                for neighbor in neighbors:
                    if not neighbor == parent:
                        pos = make_pos(pos, neighbor, currentLevel +
                                    1, node, vert_loc-vert_gap)
                return pos
            
            if levels is None:
                levels = make_levels({})
            else:
                levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
            vert_gap = height / (max([l for l in levels])+1)
            return make_pos({})


        def plot_graph(graph, pos):
            fig = plt.figure(figsize=(12, 24))
            ax = fig.gca()
            nx.draw(graph, pos=pos, with_labels=True, node_size=60, font_size=8, ax=ax)

            fig.tight_layout()
            return fig



        pos = get_pos1(G)
        fig = plot_graph(G, pos)
        # pos = get_pos2(G, root)
        # fig = plot_graph(G, pos)

        with gr.Row():
            gr.Textbox(value="Image Classficiation", label="Task")
            gr.Textbox(value=f"{self.experiment.model.__class__.__name__}", label="Model")
        gr.Plot(value=fig, label=f"Model Architecture of {self.experiment.model.__class__.__name__}", visible=True)



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
    def __init__(self, exp_info):
        self.exp_info = exp_info
        self.experiment = exp_info['experiment']
        self.input_visualizer = exp_info['input_visualizer']
        self.target_visualizer = exp_info['target_visualizer']

    def viz_input(self, input, data_id):
        orig_img_np = self.input_visualizer(input)
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
        text = f"Ground Truth Label: {self.target_visualizer(record['label'])}\n"

        for ind, pred in enumerate(probs.argsort()[-topk:][::-1]):
            label = self.target_visualizer(torch.tensor(pred))
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
        if len(record['explanations'][0]['evaluations']) > 0:
            record['explanations'] = sorted(record['explanations'], key=lambda x: x['evaluations'][0]['value'], reverse=True)

        return record


    def show(self):
        with gr.Row():
            gr.Textbox(value="Image Classficiation", label="Task")
            gr.Textbox(value=f"{self.experiment.model.__class__.__name__}", label="Model")
            gr.Textbox(value="Heatmap", label="Explanation Type")

        dset = self.experiment.manager._data.dataset
        imgs = []
        for i in range(len(dset)):
            img = self.input_visualizer(dset[i][0])
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
                cr_metrics = gr.CheckboxGroup(choices=cr_metrics_names, value=[cr_metrics_names[0]], label="Correctness")
                def on_select(metrics):
                    if cr_metrics_names[0] not in metrics:
                        gr.Warning(f"{cr_metrics_names[0]} is required for the sorting the explanations.")
                        return [cr_metrics_names[0]] + metrics
                    else:
                        return metrics

                cr_metrics.select(on_select, inputs=cr_metrics, outputs=cr_metrics)
            with gr.Row():
                # cn_metrics = gr.CheckboxGroup(choices=cn_metrics_names, value=cn_metrics_names, label="Continuity")
                cn_metrics = gr.CheckboxGroup(choices=cn_metrics_names, label="Continuity")
            with gr.Row():
                # cp_metrics = gr.CheckboxGroup(choices=cp_metrics_names, value=cp_metrics_names[0], label="Compactness")
                cp_metrics = gr.CheckboxGroup(choices=cp_metrics_names, label="Compactness")

        metric_inputs = [cr_metrics, cn_metrics, cp_metrics]

        data_id = gallery.selected_index
        bttn = gr.Button("Explain", variant="primary")

        buffer_size =  2 * len(explainer_names)
        buffer_n_rows = buffer_size // PLOT_PER_LINE
        buffer_n_rows = buffer_n_rows + 1 if buffer_size % PLOT_PER_LINE != 0 else buffer_n_rows

        plots = [gr.Textbox(label="Prediction result", visible=False)]
        for i in range(buffer_n_rows):
            with gr.Row():
                for j in range(PLOT_PER_LINE):
                    plot = gr.Image(value=None, label="Blank", visible=False)
                    plots.append(plot)

        def show_plots():
            _plots = [gr.Textbox(label="Prediction result", visible=False)]
            num_plots = sum([1 for info in self.explainer_checkbox_group.info if info['checked']])
            n_rows = num_plots // PLOT_PER_LINE
            n_rows = n_rows + 1 if num_plots % PLOT_PER_LINE != 0 else n_rows
            _plots += [gr.Image(value=None, label="Blank", visible=True)] * (n_rows * PLOT_PER_LINE)
            _plots += [gr.Image(value=None, label="Blank", visible=False)] * ((buffer_n_rows - n_rows) * PLOT_PER_LINE)
            return _plots
        
        def render_plots(data_id, *metric_inputs):
            # Clear Cache Files
            cache_dir = f"{os.environ['GRADIO_TEMP_DIR']}/res"
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            for f in os.listdir(cache_dir):
                if len(f.split(".")[0]) == 16:
                    os.remove(os.path.join(cache_dir, f))

            # Render Plots
            metric_input = []
            for metric in metric_inputs:
                if metric:
                    metric_input += metric
                    
            record = self.generate_record(data_id, metric_input)

            pred = self.get_prediction(record)
            plots = [gr.Textbox(label="Prediction result", value=pred, visible=True)]

            num_plots = sum([1 for info in self.explainer_checkbox_group.info if info['checked']])
            n_rows = num_plots // PLOT_PER_LINE
            n_rows = n_rows + 1 if num_plots % PLOT_PER_LINE != 0 else n_rows

            for i in range(n_rows):
                for j in range(PLOT_PER_LINE):
                    if i*PLOT_PER_LINE+j < len(record['explanations']):
                        exp_res = record['explanations'][i*PLOT_PER_LINE+j]
                        path = self.get_exp_plot(data_id, exp_res)
                        plot_obj = gr.Image(value=path, label=f"{exp_res['explainer_nm']} ({exp_res['mode']})", visible=True)
                        plots.append(plot_obj)
                    else:
                        plots.append(gr.Image(value=None, label="Blank", visible=True))
            
            plots += [gr.Image(value=None, label="Blank", visible=False)] * ((buffer_n_rows - n_rows) * PLOT_PER_LINE)

            return plots
        
        bttn.click(show_plots, outputs=plots)
        bttn.click(render_plots, inputs=[data_id] + metric_inputs, outputs=plots)



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
            exp_nm = exp.__class__.__name__
            if exp_nm in DEFAULT_EXPLAINER:
                checked = True
            else:
                checked = False
            self.info.append({'nm': exp_nm, 'id': exp_id, 'pp_id' : 0, 'mode': 'default', 'checked': checked})

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
        bttns = []
        for exp in self.explainer_objs:
            val = exp.explainer_name in DEFAULT_EXPLAINER
            checkboxes.append(gr.Checkbox(label="Default Parameter", value=val, interactive=True))
        checkboxes += [gr.Checkbox(label="Optimized Parameter (Not Optimal)", value=False, interactive=False)] * len(self.explainer_objs)
        bttns += [gr.Button(value="Optimize", size="sm", variant="primary")] * len(self.explainer_objs)

        for exp in self.explainer_objs:
            self.update_check(exp.default_exp_id, True)
            if hasattr(exp, "optimal_exp_id"):
                self.update_check(exp.optimal_exp_id, False)
        return checkboxes + bttns

    def get_checkboxes(self):
        checkboxes = []
        checkboxes += [exp.default_check for exp in self.explainer_objs]
        checkboxes += [exp.opt_check for exp in self.explainer_objs]
        return checkboxes
    
    def get_bttns(self):
        return [exp.bttn for exp in self.explainer_objs]
    
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
        bttns = self.get_bttns()
        self.gallery.gallery_obj.select(
            fn=self.update_gallery_change,
            outputs=checkboxes + bttns
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
        # if self.explainer_name in ["Lime", "KernelShap", "IntegratedGradients"]:
        #     gr.Info("Lime, KernelShap and IntegratedGradients currently do not support hyperparameter optimization.")
        #     return [gr.update()] * 2
        
        data_id = self.gallery.selected_index
        
        optimized, _, _ = self.experiment.optimize(
            data_id=data_id.value,
            explainer_id=self.default_exp_id,
            metric_id=self.obj_metric,
            direction='maximize',
            sampler=SAMPLE_METHOD,
            n_trials=OPT_N_TRIALS,
        )

        opt_explainer_id = optimized['explainer_id']
        opt_postprocessor_id = optimized['postprocessor_id']

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
            checked = next(filter(lambda x: x['nm'] == self.explainer_name, self.groups.info))['checked']
            self.default_check = gr.Checkbox(label="Default Parameter", value=checked, interactive=True)
            self.opt_check = gr.Checkbox(label="Optimized Parameter (Not Optimal)", interactive=False)

            self.default_check.select(self.default_on_select)
            self.opt_check.select(self.optimal_on_select)

            self.bttn = gr.Button(value="Optimize", size="sm", variant="primary")
            self.bttn.click(self.optimize, outputs=[self.opt_check, self.bttn], queue=True, concurrency_limit=1)
        

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
                x=0,
                y=-0.1 * (cnt+1),
                xref='paper',
                yref='paper',
                text=metric_text,
                showarrow=False,
                font=dict(
                    size=18,
                ),
            )
            cnt += 1


        fig = fig.update_layout(
            width=380,
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
            margin=dict(t=40, b=40*cnt, l=20, r=20),
        )

        # Generate Random Unique ID
        root = f"{os.environ['GRADIO_TEMP_DIR']}/res"
        if not os.path.exists(root): os.makedirs(root)
        key = secrets.token_hex(8)
        path = f"{root}/{key}.png"
        fig.write_image(path)
        return path


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
        <a href="https://openxaiproject.github.io/pnpxai/">
            <img src="/file=data/static/XAI-Top-PnP.svg" width="100" height="100">
        </a>
            <h1> Plug and Play XAI Platform for Image Classification </h1>
        </div>
        """

    def launch(self, **kwargs):
        with gr.Blocks(
            title=self.name,
        ) as demo:
            cwd = os.getcwd()
            gr.set_static_paths(cwd)
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

experiments = {}

model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
experiment1 = AutoExplanationForImageClassification(
    model=model,
    data=loader,
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    channel_dim=1
)

experiments['experiment1'] = {
    'name': 'ResNet18',
    'experiment': experiment1,
    'input_visualizer': lambda x: denormalize_image(x, transform.mean, transform.std),
    'target_visualizer': target_visualizer,
}


model, transform = get_torchvision_model('vit_b_16')
dataset = get_imagenet_dataset(transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
experiment2 = AutoExplanationForImageClassification(
    model=model,
    data=loader,
    input_extractor=lambda batch: batch[0],
    label_extractor=lambda batch: batch[-1],
    target_extractor=lambda outputs: outputs.argmax(-1),
    channel_dim=1
)

experiments['experiment2'] = {
    'name': 'ViT-B_16',
    'experiment': experiment2,
    'input_visualizer': lambda x: denormalize_image(x, transform.mean, transform.std),
    'target_visualizer': target_visualizer,
}

app = ImageClsApp(experiments)
demo = app.launch()
demo.launch(favicon_path="data/static/XAI-Top-PnP.svg", share=True)
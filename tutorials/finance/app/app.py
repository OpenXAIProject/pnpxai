import gradio as gr
from pnpxai import AutoExperiment
import plotly.graph_objects as go

PLOT_PER_LINE = 2
N_FEATURES_TO_SHOW = 5
NAME_MAP = {
    "TabABPC" : {"cat" : "Correctness", "alias" : "AbPC"},
    "TabMoRF" : {"cat" : "Correctness", "alias" : "MoRF"},
    "TabLeRF" : {"cat" : "Correctness", "alias" : "LeRF"},
    "TabInfidelity" : {"cat" : "Correctness", "alias" : "MuFidelity"},
    "TabAvgSensitivity" : {"cat" : "Continuity", "alias" : "Sensitivity"},
    "TabComplexity" : {"cat" : "Compactness", "alias" : "Complexity"},
}
ALIAS_MAP = {
    "AbPC" : "TabABPC",
    "MoRF" : "TabMoRF",
    "LeRF" : "TabLeRF",
    "MuFidelity" : "TabInfidelity",
    "Sensitivity" : "TabAvgSensitivity",
    "Complexity" : "TabComplexity",
}

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
                detector_res.show()

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
        pass

    def describe(self):
        return "This component shows the detection result."

    def show(self):
        self.describe()

class Database(Component):
    def __init__(self, data):
        self.data = data

    def handle_click(self, search_text):
        if search_text == "":
            filtered_df = self.data
        else:
            try:
                index = int(search_text)
                if 0 <= index < len(self.data):
                    filtered_df = self.data.loc[[index]]
                else:
                    filtered_df = pd.DataFrame(columns=self.data.columns)  # Empty DataFrame
            except ValueError:
                filtered_df = pd.DataFrame(columns=self.data.columns)  # Empty DataFrame
        return filtered_df

    def show(self):
        search_input = gr.Textbox(label="Enter index")
        df = gr.DataFrame(self.data)
        bttn = gr.Button("Search")
        bttn.click(self.handle_click, inputs=search_input, outputs=df)

class Experiment(Component):
    def __init__(self, experiment):
        self.experiment = experiment
        self.experiment.seq = 0

    def get_prediction(self, record):
        index = record['data_id']
        result = "fraud" if record['target'] == 1 else "non-fraud"
        prob = record['output'].max()
        text = f"The model predicts data {index} as {result} with probability {prob*100:.2f}%"
        return text
    
    def get_exp_plot(self, data_index, exp_res):
        return ExpRes(data_index, exp_res).show()

    def handle_click(self, data_id, explainer_names, *metric_nm_inputs):
        metric_names = []
        for metric_nm_input in metric_nm_inputs:
            metric_names.extend(metric_nm_input)
        
        metric_names = [ALIAS_MAP[name] if name in ALIAS_MAP else name for name in metric_names]

        all_explainers = self.experiment.manager.all_explainers
        all_explainer_ids = list(range(len(all_explainers)))
        all_metrics = self.experiment.manager.all_metrics
        all_metric_ids = list(range(len(all_metrics)))
        # all_explainers, all_explainer_ids = self.experiment.manager.get_explainers()
        # all_metrics, all_metric_ids = self.experiment.manager.get_metrics()
        all_explainer_names = [exp.__class__.__name__ for exp in all_explainers]
        all_metric_names = [metric.__class__.__name__ for metric in all_metrics]

        explainer_ids = [all_explainer_ids[all_explainer_names.index(name)] for name in explainer_names]
        metric_ids = [all_metric_ids[all_metric_names.index(name)] for name in metric_names]
        self.experiment.run(
            data_ids=[data_id],
            explainer_ids=explainer_ids, 
            metrics_ids=metric_ids
        )

        record = self.experiment.records[0]
        pred = self.get_prediction(record)
        plots = []

        # Sort the explanations with its AbPC value
        explanation = record['explanations']
        for exp in explanation:
            exp['sorting_value'] = next(filter(lambda x: x['metric_nm'] == 'TabABPC', exp['evaluations']))['value']
            if exp['sorting_value'] is None:
                raise ValueError("AbPC value is not found in the evaluations")
        explanation = sorted(explanation, key=lambda x: x['sorting_value'], reverse=True)
        for exp_res in explanation:
            plots.append(self.get_exp_plot(data_id, exp_res))

        if len(record['explanations']) < len(all_explainers):
            for _ in range(len(all_explainers) - len(record['explanations'])):
                plots.append(None)

        return [pred] + plots

    def show(self):
        gr.Label(f"Experiment ({self.experiment.model.__class__.__name__})")

        dset = self.experiment.manager._data.dataset
        if isinstance(dset, NumpyDataset):
            data = dset.inputs
        elif isinstance(dset, torch.utils.data.TensorDataset):
            data = dset.tensors[0].numpy()
        else:
            raise ValueError("Unsupported dataset type")
        
        database = Database(data)
        database.show()
        data_index = gr.Number(label="Data index")
        
        explainers, _ = self.experiment.manager.get_explainers()
        explainer_names = [exp.__class__.__name__ for exp in explainers]
        explainer_input = gr.CheckboxGroup(label="Explainers", choices=explainer_names, value=explainer_names)

        metrics, _ = self.experiment.manager.get_metrics()
        metrics_names = [metric.__class__.__name__ for metric in metrics]

        correctness_metrics = [NAME_MAP[name]["alias"] for name in metrics_names if name in NAME_MAP and NAME_MAP[name]["cat"] == "Correctness"]
        continuity_metrics = [NAME_MAP[name]["alias"] for name in metrics_names if name in NAME_MAP and NAME_MAP[name]["cat"] == "Continuity"]
        compactness_metrics = [NAME_MAP[name]["alias"] for name in metrics_names if name in NAME_MAP and NAME_MAP[name]["cat"] == "Compactness"]


        with gr.Accordion(label="Evaluators", open=True):
            with gr.Row():
                with gr.Column():
                    cor_input = gr.CheckboxGroup(label="Correctness", choices=correctness_metrics, value=correctness_metrics)
                with gr.Column():
                    con_input = gr.CheckboxGroup(label="Continuity", choices=continuity_metrics, value=continuity_metrics)
                with gr.Column():
                    com_input = gr.CheckboxGroup(label="Compactness", choices=compactness_metrics, value=compactness_metrics)

        bttn = gr.Button("Explain")

        with gr.Row():
            prediction_result = gr.Textbox(label="Prediction result")
        
        n_rows = len(explainers) // PLOT_PER_LINE
        plots = []
        for i in range(n_rows):
            with gr.Row():
                for j in range(PLOT_PER_LINE):
                    plots.append(gr.Plot(label=f"Explanation {1+i*PLOT_PER_LINE+j}"))

        bttn.click(
            self.handle_click, 
            inputs=[
                data_index, 
                explainer_input,
                cor_input,
                con_input,
                com_input,
            ], 
            outputs=[prediction_result] + plots
        )
        

class ExpRes(Component):
    def __init__(self, data_index, exp_res):
        self.data_index = data_index
        self.exp_res = exp_res

    def show(self):
        value = self.exp_res['value']
        explainer_nm = self.exp_res['explainer_nm']
        index = np.arange(value.shape[0])

        fig = go.Figure(data=[go.Bar(x=index, y=value, name='Attribution')])

        evaluations = self.exp_res['evaluations']
        metric_values = [f"{NAME_MAP[eval['metric_nm']]['alias']}: {eval['value']:.2f}" for eval in evaluations]
        metric_text = ', '.join(metric_values)

        # Add annotation with metric values
        fig.add_annotation(
            x=0,
            y=-0.3,
            xref='paper',
            yref='paper',
            text=metric_text,
            showarrow=False,
            font=dict(
                size=12,
            ),
        )

        fig.update_layout(
            title=f"{explainer_nm}",
            width=700,
            height=400,
            xaxis=dict(title='Index'),
            yaxis=dict(title='Attribution'),
            margin=dict(t=30, b=200),
        )

        return fig


class FinanceApp(App):
    def __init__(self, experiments, **kwargs):
        self.name = "Finance App"
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

if __name__ == '__main__':
    import pickle
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from model import TabResNet


    with open("data/baf/preprocess/X_test.npy", 'rb') as f:
        preprocessed_data = np.load(f)

    with open("data/baf/preprocess/X_train.npy", 'rb') as f:
        X_train = np.load(f)

    with open("data/baf/preprocess/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    class NumpyDataset(Dataset):
        def __init__(self, inputs: np.ndarray):
            super().__init__()
            self.inputs = inputs

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx]

    np_dataset = NumpyDataset(inputs=preprocessed_data)
    np_loader = DataLoader(np_dataset, batch_size=8, shuffle=False)
    th_dataset = TensorDataset(torch.tensor(preprocessed_data).float())
    torch_loader = DataLoader(th_dataset, batch_size=8, shuffle=False)

    with open("models/baf/model_lr.pkl", 'rb') as f:
        model_lr = pickle.load(f)

    with open("models/baf/model_rf.pkl", 'rb') as f:
        model_rf = pickle.load(f)

    with open("models/baf/model_xgb.pkl", 'rb') as f:
        model_xgb = pickle.load(f)

    model_nn = TabResNet(preprocessed_data.shape[1], 2)
    model_nn.load_state_dict(torch.load("models/baf/tabresnet.pth"))
    model_nn = torch.nn.Sequential(model_nn, torch.nn.Softmax(dim=1))
    model_nn.eval()

    # Abnormal data indices
    # [ 3,  5,  6,  7,  8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23,
    #    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43,
    #    45, 46, 49, 50, 51, 55]

    experiments = []

    experiment1 = AutoExperiment(
        model=model_lr,
        data=np_loader,
        modality='tabular',
        question='why',
        evaluator_enabled=True,
        input_extractor=lambda batch: batch,
        label_extractor=lambda batch: torch.zeros(len(batch), dtype=torch.long),
        target_extractor=lambda outputs: outputs.argmax(-1),
        background_data=X_train,
    )

    experiment2 = AutoExperiment(
        model=model_nn,
        data=torch_loader,
        modality='tabular',
        question='why',
        evaluator_enabled=True,
        input_extractor=lambda batch: batch,
        label_extractor=lambda batch: torch.zeros(len(batch), dtype=torch.long),
        target_extractor=lambda outputs: outputs.argmax(-1),
        background_data=torch.tensor(X_train).float(),
    )

    experiments.append(experiment1)
    experiments.append(experiment2)

    app = FinanceApp(experiments)

    with gr.Blocks(
            title=app.name,
        ) as demo:
            app.overview_tab.show()
            app.detection_tab.show()
            app.local_exp_tab.show()

    demo.launch(share=True)
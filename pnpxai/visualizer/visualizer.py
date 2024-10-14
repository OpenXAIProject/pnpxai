import numpy as np
from collections import defaultdict
from typing import Union, Dict, List, Optional, Tuple, Sequence

import gradio as gr
from plotly import express as px
from matplotlib import pyplot as plt

from pnpxai import Experiment
from pnpxai.explainers import Explainer
from pnpxai.evaluator.metrics import Metric
from pnpxai.explainers.utils import PostProcessor


class VisualizerInterface:
    def __init__(self):
        self.blocks: gr.Blocks = None

    def build(
        self,
        inputs: List[np.ndarray],
        explainers: Optional[List[Union[Explainer, Tuple[Explainer, int]]]] = None,
        metrics: Optional[List[Union[Metric, Tuple[Metric, int]]]] = None,
        postprocessors: Optional[
            List[Union[PostProcessor, Tuple[PostProcessor, int]]]
        ] = None,
        on_options_change: Optional[callable] = None,
    ):
        with gr.Blocks() as self.blocks:
            btn_toggle_sidebar = gr.Button("Toggle Options")
            state_options = gr.State(True)
            state_input_id = gr.State(None)
            state_data = gr.State([])

            with gr.Row():
                # Setup column
                with gr.Column(visible=True) as col_sidebar:
                    # Explainers
                    select_explainers = gr.Dropdown(
                        label="Explainers",
                        choices=explainers,
                        multiselect=True,
                        show_label=True,
                    )
                    select_metrics = gr.Dropdown(
                        label="Metrics",
                        choices=metrics,
                        multiselect=True,
                        show_label=True,
                    )
                    select_postprocessors = gr.Dropdown(
                        label="PostProcessors",
                        choices=postprocessors,
                        multiselect=False,
                        show_label=True,
                    )
                    btn_options_change = gr.Button("Submit")

                btn_toggle_sidebar.click(
                    self._toggle_options, [state_options], [col_sidebar, state_options]
                )

                # Data column
                with gr.Column(scale=2) as col_main:
                    # Inputs
                    # Outputs w/ explanations
                    gallery_input = gr.Gallery(value=inputs, label="Inputs", inputs=0, height=400)
                    gallery_input.select(self._set_input_id, None, [state_input_id])

                    @gr.render(inputs=state_data)
                    def render_outputs(outputs):
                        if len(outputs) >= 0:
                            with gr.Row():
                                for explainer_id in outputs:
                                    self._plot_output_column(outputs[explainer_id])

            if on_options_change is not None:
                btn_options_change.click(
                    on_options_change,
                    [
                        state_input_id,
                        select_explainers,
                        select_metrics,
                        select_postprocessors,
                    ],
                    [state_data],
                )

            pass

    def _plot_output_column(self, explainer_data: Dict[int, dict]):
        with gr.Column():
            if len(explainer_data) > 0:
                datum = next(iter(explainer_data.values()))
                gr.Markdown(f"{datum['explainer'].__class__.__name__}")
                explanation = datum["postprocessed"].cpu().detach().squeeze().numpy()
                exp_plot = plt.figure()
                axes = exp_plot.subplots(1, 1)
                axes.imshow(explanation, cmap="twilight")

                plot_title = "\n".join(
                    [
                        f"{explainer_data[metric_id]['metric'].__class__.__name__}: {explainer_data[metric_id]['evaluation'].item():.2f}"
                        for metric_id in explainer_data
                    ]
                )
                gr.Plot(exp_plot, label=plot_title, show_label=True)
            else:
                gr.Markdown("Something went wrong!")

    def _toggle_options(self, state: gr.State):
        state = not state
        return gr.update(visible=state), state

    def _set_input_id(self, evt: gr.SelectData):
        return evt.index

    def launch(self):
        self.blocks.launch()


class Visualizer:
    def __init__(
        self,
        experiment: Experiment,
        input_visualizer: Optional[callable] = None,
    ):
        self.experiment = experiment
        self._vi: VisualizerInterface = None
        self._input_visualizer = input_visualizer
        self.build()

    def build(self):
        self._vi = VisualizerInterface()
        self._vi.build(
            self._get_input_data(),
            explainers=self._get_explainers_options(),
            metrics=self._get_metrics_options(),
            postprocessors=self._get_postprocessors_options(),
            on_options_change=self._on_options_change,
        )

    def _get_explainers_options(self) -> List[Tuple[int, str]]:
        return list(zip(*self.experiment.manager.get_explainers()))

    def _get_metrics_options(self) -> List[Tuple[int, str]]:
        return list(zip(*self.experiment.manager.get_metrics()))

    def _get_postprocessors_options(self) -> List[Tuple[int, str]]:
        return list(zip(*self.experiment.manager.get_postprocessors()))

    def _get_input_data(self) -> List[np.ndarray]:
        return [
            (
                self._input_visualizer(datum)
                if self._input_visualizer is not None
                else datum
            )
            for datum in self.experiment.get_all_inputs_flattened()
        ]

    def _on_options_change(
        self,
        data_id: int,
        explainer_ids: Sequence[int],
        metric_ids: Sequence[int],
        postprocessor_id: int,
    ):
        outputs = defaultdict(dict)
        # Convert data_id from range to actual data_id
        data_id = self.experiment.manager.get_data()[1][data_id]
        for explainer_id in explainer_ids:
            for metric_id in metric_ids:
                outputs[explainer_id][metric_id] = self.experiment.run_batch(
                    data_ids=[data_id],
                    explainer_id=explainer_id,
                    metric_id=metric_id,
                    postprocessor_id=postprocessor_id,
                )
        return outputs

    def launch(self):
        self._vi.launch()

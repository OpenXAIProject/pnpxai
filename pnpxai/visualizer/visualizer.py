import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Sequence

import gradio as gr
from matplotlib import pyplot as plt

from pnpxai import Experiment
from pnpxai.utils import format_into_tuple


class VisualizerInterface:
    def __init__(self):
        self.blocks: gr.Blocks = None

    def build(
        self,
        inputs: List[np.ndarray],
        explainer_options: Optional[Tuple[str, str]] = None,
        metric_options: Optional[Tuple[str, str]] = None,
        pooling_fn_options: Optional[Tuple[str, str]] = None,
        normalization_fn_options: Optional[Tuple[str, str]] = None,
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
                        choices=explainer_options,
                        multiselect=True,
                        show_label=True,
                    )
                    select_metrics = gr.Dropdown(
                        label="Metrics",
                        choices=metric_options,
                        multiselect=True,
                        show_label=True,
                    )
                    select_pooling_fn = gr.Dropdown(
                        label='Pooling Methods',
                        choices=pooling_fn_options,
                        multiselect=False,
                        show_label=True,
                    )
                    select_norm_fn = gr.Dropdown(
                        label='Normalization Methods',
                        choices=normalization_fn_options,
                        multiselect=False,
                        show_label=True,
                    )
                    # select_postprocessors = gr.Dropdown(
                    #     label="PostProcessors",
                    #     choices=postprocessors,
                    #     multiselect=False,
                    #     show_label=True,
                    # )
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
                                for explainer_key in outputs:
                                    self._plot_output_column(outputs[explainer_key])
            if on_options_change is not None:
                btn_options_change.click(
                    on_options_change,
                    [
                        state_input_id,
                        select_explainers,
                        select_metrics,
                        select_pooling_fn,
                        select_norm_fn,
                    ],
                    [state_data],
                )

            pass

    def _plot_output_column(self, explainer_data: Dict[int, dict]):
        with gr.Column():
            if len(explainer_data) > 0:
                datum = next(iter(explainer_data.values()))
                gr.Markdown(f"{datum.explainer.__class__.__name__}")
                explanation = datum.explanations.cpu().detach().squeeze().numpy()
                exp_plot = plt.figure()
                axes = exp_plot.subplots(1, 1)
                axes.imshow(explanation, cmap="twilight")

                plot_title = "\n".join(
                    [
                        f"{explainer_data[metric_key].metric.__class__.__name__}: {explainer_data[metric_key].evaluations.item():.2f}"
                        for metric_key in explainer_data
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

    def launch(self, share=False):
        self.blocks.launch(share=share)


class Visualizer:
    def __init__(
        self,
        experiment: Experiment,
        input_visualizer: Optional[callable] = None,
    ):
        assert len(format_into_tuple(experiment.modality)) == 1, 'Multimodal not supported'
        self.experiment = experiment
        self._vi: VisualizerInterface = None
        self._input_visualizer = input_visualizer
        self.build()

    def build(self):
        self._vi = VisualizerInterface()
        self._vi.build(
            self._get_input_data(),
            explainer_options=self._get_options(self.experiment.explainers),
            metric_options=self._get_options(self.experiment.metrics),
            pooling_fn_options=self._get_options(
                self.experiment.modality.util_functions['pooling_fn']),
            normalization_fn_options=self._get_options(
                self.experiment.modality.util_functions['normalization_fn']),
            on_options_change=self._on_options_change,
        )

    def _get_options(self, selector):
        return [(v.__name__, k) for k, v in selector.data.items()]

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
        explainer_keys: Sequence[int],
        metric_keys: Sequence[int],
        pooling_method,
        normalization_method,
        # postprocessor_id: int,
    ):
        outputs = defaultdict(dict)
        # Convert data_id from range to actual data_id
        data_id = self.experiment.manager.get_data()[1][data_id]
        for explainer_key in explainer_keys:
            for metric_key in metric_keys:
                outputs[explainer_key][metric_key] = self.experiment.run_batch(
                    explainer_key=explainer_key,
                    metric_key=metric_key,
                    pooling_method=pooling_method,
                    normalization_method=normalization_method,
                    data_ids=[data_id],
                )
        return outputs

    def launch(self, share=False):
        self._vi.launch(share=share)

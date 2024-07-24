import os
import gradio as gr
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
import joblib

import pickle
from model import TabResNet
import torch
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Assuming you have these explainers implemented
from pnpxai.explainers import KernelShap, Lime, LRP, IntegratedGradients, TabKernelShap, TabLime

demo = None

class XAIApp:
    def __init__(self, raw_data: pd.DataFrame, preprocessed_data: pd.DataFrame, 
                 metadata: Dict[str, Any], trained_models: Dict[str, Any],
                 **kwargs
                 ):
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.metadata = metadata
        self.trained_models = trained_models
        self.bg_data = kwargs.get("bg_data", None)
        with open("evaluations.pkl", "rb") as f:
            self.evaluations = pickle.load(f)

        self.sample_for_global = kwargs.get("sample_for_global", 10)
        self.k_input = kwargs.get("k_input", 10)
        self.eval_metrics = kwargs.get("eval_metrics", ['MoRF/LeRF', 'Infidelity', 'AvgSensitivity', 'Complexity'])

        self.theme = kwargs.get("theme", gr.Theme.load("theme.json"))
        self.css = kwargs.get("css", "")
        self.js = kwargs.get("js", "")
        self.head = kwargs.get("head", "")


    def show_title(self) -> str:
        return """
        <div style="text-align: center;">
            <img src="/file=static/XAI-Top-PnP.svg" width="100" height="100">
            <h1> 은행 계좌 개설 시 사기 계좌 탐지 모델 설명 플랫폼 (Plug and Play) </h1>
            <p>이 플랫폼은 은행 계좌 개설 시 사기 행위를 탐지하는 모델을 설명하는 데 사용됩니다.</p>
        </div>
        """

    def show_project_description(self) -> str:
        return """
        <h1>프로젝트 개요</h1>
        <p>이 프로젝트는 은행 계좌 개설 과정에서 사기 행위를 탐지하는 모델을 Plug and Play 방식으로 설명하는 방식을 제공합니다  </p>
        <p>우리는 사기 탐지 모델에 대한 통찰을 제공하기 위해 다양한 설명 가능한 AI (XAI) 방법을 활용합니다.</p>
        <p>우리의 도구는 주어진 모델에 다양한 XAI 기법을 쉽게 적용할 수 있게 해주며,
        자동으로 모델 유형을 감지하고, 사용 가능한 설명자를 추천하며, 설명 결과를 제시합니다.</p>
        <p>본 프로젝트의 주요 특징:</p>
        <ul>
            <li>다양한 사기 탐지 모델 지원</li>
            <li>다양한 설명 가능한 AI (XAI) 기법 적용 가능</li>
            <li>모델 성능 평가 및 비교</li>
            <li>사용자 친화적인 인터페이스 제공</li>
        </ul>
        <p>본 프로젝트의 궁극적인 목표는 은행 계좌 개설 시 발생할 수 있는 잠재적 사기 행위를 미리 탐지하여, 
        고객과 은행 모두의 피해를 최소화하는 것입니다.</p>
        """

    def show_data_model_info(self) -> str:
        model_info = f"탐지된 모델: {', '.join(self.trained_models.keys())}"
        
        raw_info = f"""원본 데이터의 길이는 {self.raw_data.shape[0]}이며, 각 데이터는 다음과 같은 특성을 가지고 있습니다: {', '.join(self.raw_data.columns)}
        이 중 수치형 데이터는 {len(self.metadata['float_cols'])}개이며, 범주형 데이터는 {len(self.metadata['cat_cols'])}개가 있습니다.
        """
        
        preprocessed_info = f"전처리된 데이터의 길이는 {self.preprocessed_data.shape[0]}이며, 각 데이터는 {self.preprocessed_data.shape[1]}개의 특성을 가진 데이터로 변환되었습니다."
        return f"""
        <div style='margin: 20px;'>
            <h2> 모델 및 데이터 정보</h2>
            <h3> 탐지된 모델 종류 </h3>
            <p>{model_info}</p>
            <h3> 데이터 정보 </h3>
            <p>{raw_info}</p>
            <h3> 전처리된 데이터 정보 </h3>
            <p>{preprocessed_info}</p>
        </div>
        """
    
    def show_global_explanation(self, model_name: str = None) -> str:

        if model_name is None:
            return """
            <div>
            <h1> 전역적 설명 (Global Explanation)</h1>
            <ul>
                <li><strong>특성 영향력:</strong> 그래프의 y축은 다양한 특성들을 나열하고 있으며, x축은 각 특성의 SHAP 값을 나타냅니다. SHAP 값이 0에서 멀어질수록 해당 특성의 영향력이 큽니다.</li>
            </ul>
            """
        else:
            return f"""
            <div class="description">
                <h2> {model_name} 모델에 대한 Global Explanation </h1>
                <ul>
                    <li><strong>상위 영향 특성:</strong> 'income', 'name_email_similarity', 'prev_address_months_count' 등이 모델 예측에 큰 영향을 미치는 것으로 보입니다.</li>
                    <li><strong>양방향 영향:</strong> 일부 특성들은 양과 음의 SHAP 값을 모두 가지고 있어, 상황에 따라 예측에 긍정적 또는 부정적 영향을 줄 수 있음을 보여줍니다.</li>
                    <li><strong>타겟 분포:</strong> 점의 색상은 타겟 값(0.2부터 0.8까지)을 나타내며, 이는 모델이 예측하려는 결과값의 범위를 보여줍니다.</li>
                    <li><strong>비선형성:</strong> 일부 특성들(예: 'income')은 넓은 feature importance 값 범위를 가지고 있어, 해당 특성이 비선형적으로 예측에 영향을 미칠 수 있음을 시사합니다.</li>
                    <li><strong>이진 특성:</strong> 'email_is_free', 'phone_home_valid' 같은 특성들은 뚜렷한 이진 패턴을 보여, 범주형 변수일 가능성이 높습니다.</li>
                    <li><strong>모델 복잡성:</strong> 많은 특성들이 다양한 SHAP 값을 가지고 있어, 이 모델이 상당히 복잡하고 다양한 요인들을 고려하고 있음을 알 수 있습니다.</li>
                </ul>
            </div>
            """

    def show_evaluation_metric_decription(self) -> str:
        return """    
        <div class="metric">
            <h2>Infidelity</h2>
            <p>Yeh et al., 2019에 의해 구현된 Infidelity 메트릭입니다.</p>
            <p>설명의 불충실성은 1) 설명과 입력 섭동의 내적곱과 2) 중요한 섭동 후 모델 출력의 차이 사이의 예상 평균 제곱 오차를 나타냅니다.</p>
            <p><strong>참조:</strong></p>
            <ul>
                <li>Chih-Kuan Yeh et al.: "On the (In)fidelity and Sensitivity of Explanations." 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.</li>
            </ul>
        </div>

        <div class="metric">
            <h2>AvgSensitivity</h2>
            <p>Yeh et al., 2019에 의해 구현된 Avg-Sensitivity 메트릭입니다.</p>
            <p>몬테카를로 샘플링 기반 근사를 사용하여 설명이 약간의 섭동 하에서 어떻게 변화하는지를 측정합니다 - 평균 민감도가 포착됩니다.</p>
            <p><strong>참조:</strong></p>
            <ul>
                <li>Chih-Kuan Yeh et al. "On the (in)fidelity and sensitivity for explanations." NeurIPS (2019): 10965-10976.</li>
                <li>Umang Bhatt et al.: "Evaluating and aggregating feature-based model explanations." IJCAI (2020): 3016-3022.</li>
            </ul>
        </div>

        <div class="metric">
            <h2>Complexity</h2>
            <p>Bhatt et al., 2020에 의해 구현된 Complexity 메트릭입니다.</p>
            <p>설명의 복잡성은 속성의 총 크기에 대한 특징 x_i의 분수 기여도의 엔트로피로 정의됩니다. 복잡한 설명은 일부 결정을 설명하기 위해 설명에서 모든 특징을 사용하는 설명입니다. 이러한 설명이 모델 출력에 충실할 수 있지만, 특징의 수가 너무 많으면 사용자가 설명을 이해하기 어렵게 되어 무용지물이 될 수 있습니다.</p>
            <p><strong>참조:</strong></p>
            <ul>
                <li>Umang Bhatt et al.: "Evaluating and aggregating feature-based model explanations." IJCAI (2020): 3016-3022.</li>
            </ul>
        </div>

        <div class="metric">
            <h2>MoRF/LeRF</h2>
            <p>Samek et al., 2015에 의해 구현된 MoRF/LeRF 메트릭입니다.</p>
            <p>탐욕스러운 반복 절차를 고려하여, 이미지 x에서 정보가 점진적으로 제거될 때 이미지에 인코딩된 클래스가 어떻게 사라지는지를 측정하는 과정(지정된 위치에서의 영역 섭동)으로 구성됩니다.</p>
            <p><strong>가정:</strong></p>
            <ul>
                <li>원래 메트릭 정의는 이미지 패치 기능에 의존합니다. 따라서 메트릭은 3차원(이미지) 데이터에만 적용합니다. 다른 데이터 도메인으로 확장하려면 현재 구현에 대한 조정이 필요할 수 있습니다.</li>
            </ul>
            <p><strong>참조:</strong></p>
            <ul>
                <li>Wojciech Samek et al.: "Evaluating the visualization of what a deep neural network has learned." IEEE transactions on neural networks and learning systems 28.11 (2016): 2660-2673.</li>
            </ul>
        </div>
        """

    def detect_model_type(self, model_name: str) -> str:
        model = self.trained_models[model_name]
        
        if isinstance(model, torch.nn.Module):
            return "Neural Network"
        elif "sklearn" in str(type(model)):
            return model.__class__.__name__
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def recommend_explainers(self, model_type: str) -> List[str]:
        explainers = ["KernelSHAP", "LIME"]
        if model_type == "Neural Network":
            return explainers + ["LRP", "IG"]
        elif model_type != "Unknown":
            return ["Tab" + exp for exp in explainers]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_prediction_text(self, model_name: str, index: int) -> str:
        model_type = self.detect_model_type(model_name)
        if model_type == "Neural Network":
            data = torch.tensor(self.preprocessed_data[index]).unsqueeze(dim=0).float()
            model = self.trained_models[model_name]
            probs = model(data).softmax(dim=1)
            pred_int = probs.argmax(dim=1).item()
            pred_str = "사기 계좌" if pred_int == 1 else "정상 계좌"
        else:
            data = self.preprocessed_data[index].reshape(1, -1)
            model = self.trained_models[model_name]
            probs = model.predict_proba(data)
            pred_int = probs.argmax(axis=1).item()
            pred_str = "사기 계좌" if pred_int == 1 else "정상 계좌"

        return f"""
        주어진 {index}번 데이터에 대해 {model_name} 모델은 해당 데이터가 "{pred_str}"로 예측했으며, 이에 대한 확률은 {probs.max().item()*100:.2f}%입니다.
        """
    
    def attr_agg(self, attr):
        feature_names = self.metadata["float_cols"].tolist() + list(self.metadata["cat_cols"].keys())
        aggr = pd.DataFrame(index=np.arange(attr.shape[0]), columns=feature_names)
        aggr.loc[:, self.metadata["float_cols"]] = attr[:, :len(self.metadata["float_cols"])]
        start = len(self.metadata["float_cols"])
        for col in self.metadata['cat_cols'].keys():
            aggr.loc[:, col] = attr[:, start:start + len(self.metadata['cat_cols'][col])].sum(axis=1)
        return aggr
    
    def generate_explanation_single(self, model_name: str, data: Union[torch.Tensor, np.ndarray], explainer_name: str) -> go.Figure:
        k = self.k_input
        model = self.trained_models[model_name]
        model_type = self.detect_model_type(model_name)
        
        if model_type == "Neural Network":
            targets = model(data)
        else:
            targets = model.predict_proba(data)
        
        # Generate explanation using the specified explainer
        if explainer_name == "KernelSHAP":
            explainer = KernelShap(model)
            baselines = torch.zeros(data.shape[1])
            attr = explainer.attribute(
                inputs=data,
                targets=targets.argmax(dim=1),
                baselines=baselines,
                n_samples=400,
                feature_mask=torch.arange(data.shape[1]).unsqueeze(0),
            )

        elif explainer_name == "LIME":
            explainer = Lime(model)
            baselines = torch.zeros(data.shape[1])
            attr = explainer.attribute(
                inputs=data,
                targets=targets.argmax(dim=1),
                baselines=baselines,
                n_samples=400,
                feature_mask=torch.arange(data.shape[1]).unsqueeze(0),
            )

        elif explainer_name == "TabKernelSHAP":
            bg_data = TabKernelShap.kmeans(self.bg_data, 100)
            # bg_data = np.zeros((1, data.shape[1]))
            explainer = TabKernelShap(model, bg_data, mode="classification")
            targets = model.predict(data)
            attr = explainer.attribute(
                data,
                targets=targets,
                n_samples=400
            )

        elif explainer_name == "TabLIME":
            categorical_features = np.argwhere(np.array([len(set(self.bg_data[:,x])) for x in range(self.bg_data.shape[1])]) <= 10).flatten()
            explainer = TabLime(
                model, self.bg_data, categorical_features=categorical_features, mode='classification'
            )
            attr = explainer.attribute(
                data,
                targets=None,
                n_samples=400
            )
            attr = np.array(attr)

        elif explainer_name == "LRP":
            explainer = LRP(model)
            attr = explainer.attribute(
                inputs=data,
                targets=targets.argmax(dim=1),
            )
        elif explainer_name == "IG":
            explainer = IntegratedGradients(model)
            attr = explainer.attribute(
                inputs=data,
                targets=targets.argmax(dim=1),
            )
        else:
            raise ValueError(f"Unknown explainer: {explainer_name}")

        # Sort features by absolute value and get top k
        explanation = self.attr_agg(attr)
        cols = np.abs(explanation).T.sort_values(0).T.columns
        features, values = cols[-k:], explanation[cols[-k:]].values[0]

        # Create bar chart using Plotly
        fig = go.Figure(data=[go.Bar(x=features, y=values)])
        fig.update_layout(title=f"Top {k} features for {explainer_name}")
        return fig

    def generate_explanation(self, model_name: str, index: int, explainer_names: List[str]) -> go.Figure:
        explanation_figs = {}
        model = self.trained_models[model_name]
        model_type = self.detect_model_type(model_name)
        if model_type == "Neural Network":
            data = torch.tensor(self.preprocessed_data[index]).unsqueeze(dim=0).float()
        else:
            data = self.preprocessed_data[index].reshape(1, -1)
    
        for j in range(len(explainer_names)):
            explainer_name = explainer_names[j]
            fig = self.generate_explanation_single(model_name, data, explainer_name)
            explanation_figs[f"{model_name}_{explainer_name}"] = fig

        return explanation_figs
    
    def create_global_explanation(self, model_name: str) -> str:
        path = f"explanations/{model_name}"
        model = self.trained_models[model_name]
        model_type = self.detect_model_type(model_name)
        if model_type == "Neural Network":
            explainer = KernelShap(model)
            data = torch.tensor(self.preprocessed_data[:self.sample_for_global]).float()
            target = model(data)
            baselines = torch.zeros(data.shape[1])
            attr = explainer.attribute(
                inputs=data,
                targets=target.argmax(dim=1),
                baselines=baselines,
                n_samples=400,
                feature_mask=torch.arange(data.shape[1]).unsqueeze(0),
            ).detach().numpy()
            target = target.detach().numpy()
        else:
            bg_data = TabKernelShap.kmeans(self.bg_data, 100)
            explainer = TabKernelShap(model, bg_data, mode="classification")
            data = self.preprocessed_data[:self.sample_for_global]
            target = model.predict_proba(data)
            attr = explainer.attribute(
                data,
                targets=target.argmax(axis=1),
                n_samples=400
            )

        explanation = self.attr_agg(attr)

        if not os.path.exists(path):
            os.makedirs(path)

        explanation.to_csv(path + "/shap_global.csv")
        np.save(path + "/targets.npy", target)

        return f"{model_name} 모델에 대한 전역 설명이 생성되었습니다."

    
    def load_global_explanation(self, model_name: str) -> Union[go.Figure, plt.Figure]:
        path = f"explanations/{model_name}"
        if os.path.exists(path):
            explanation = pd.read_csv(path + "/shap_global.csv", index_col=0)
            target = np.load(path + "/targets.npy")
        else:
            return None
            
            
        def beeswarm_plot(data, title):
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_title(title)

            # Melt the DataFrame
            df_melted = data.melt(var_name='Feature', value_name='Value')

            # Convert values to numeric, coercing errors
            df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

            # Add the 'target' column back to the melted DataFrame
            df_melted['target'] = np.tile(target[:, 1], data.shape[1])

            sns.swarmplot(
                ax=ax,
                data=df_melted,
                x='Value',
                y='Feature',
                hue='target',
                orient="h", size=5, palette="coolwarm", warn_thresh=0.9
            )

            ax.grid()
            ax.set_xlabel('SHAP Value')
            ax.set_ylabel('Feature')

            fig.tight_layout()
            return fig
        
        fig = beeswarm_plot(explanation, f"Global explanation {model_name}")
        return fig

        

    def show_evaluation_metrics(self, model_name: str, eval_metric: str) -> go.Figure:
        if eval_metric == "MoRF/LeRF":
            return self.show_morf_lerf(model_name)

        elif eval_metric in ["Infidelity", "AvgSensitivity", "Complexity"]:
            return self.show_evaluation_metrics_single(model_name, eval_metric)
        
    def show_morf_lerf(self, model_name: str) -> plt.Figure:
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{self.evaluations['MoRF'][i]['explainer']} (ABPC: {self.calculate_abpc(i):.2f})" for i in range(4)])

        # Add plots to the grid
        for idx in range(len(self.evaluations["MoRF"])):
            row, col = idx // 2 + 1, idx % 2 + 1
            morf = -np.array(self.evaluations["MoRF"][idx]['value']).mean(axis=0)
            lerf = -np.array(self.evaluations["LeRF"][idx]['value']).mean(axis=0)
            morf = np.concatenate([np.array([0]), morf])
            lerf = np.concatenate([np.array([0]), lerf])
            
            fig.add_trace(go.Scatter(x=list(range(len(morf))), y=morf, mode='lines', name='MoRF'), row=row, col=col)
            fig.add_trace(go.Scatter(x=list(range(len(lerf))), y=lerf, mode='lines', name='LeRF'), row=row, col=col)

            fig.update_xaxes(title_text="Number of features", row=row, col=col)
            fig.update_yaxes(title_text="Change of the prediction value", row=row, col=col)
        
        # Update the layout of the figure
        fig.update_layout(title_text="Comparison of MoRF and LeRF", height=800, width=1200, showlegend=True)
        return fig
    
    def show_evaluation_metrics_single(self, model_name: str, eval_metric: str) -> go.Figure:
        fig = go.Figure()
        aggregation = []

        for i, obj in enumerate(self.evaluations[eval_metric]):
            explainer = obj['explainer']
            value = obj['value']
            mean_value = np.mean(value)
            
            aggregation.append({
                "explainer": explainer,
                "evaluator": eval_metric,
                "value": mean_value
            })
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=list(range(len(value))),
                y=value,
                mode='markers',
                name=f"{explainer} (mean: {mean_value:.2f})"
            ))
            
            # Horizontal line for mean value
            fig.add_trace(go.Scatter(
                x=[0, len(value) - 1],
                y=[mean_value, mean_value],
                mode='lines',
                line=dict(dash='dash'),
                name=f"{explainer} mean",
                showlegend=False
            ))
        
        fig.update_layout(
            title=eval_metric,
            xaxis_title="Index",
            yaxis_title="Value",
            legend_title="Explainer",
            height=400,
            width=1400,
            template="plotly_white"
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        return fig


    def calculate_abpc(self, idx) -> float:
        morf = -np.array(self.evaluations["MoRF"][idx]['value']).mean(axis=0)
        lerf = -np.array(self.evaluations["LeRF"][idx]['value']).mean(axis=0)
        morf = np.concatenate([np.array([0]), morf])
        lerf = np.concatenate([np.array([0]), lerf])
        return (lerf - morf).sum()

    # def provide_explanation(self, model_name: str, index: int, explanation: pd.DataFrame, prediction: int, prob: float) -> str:
    #     # Index of explanation : Explainer
    #     # Columns of explanation : Features
    #     return f"""
    #     주어진 {index}번 데이터에 대한 모델 {model_name}의 예측은 {prediction}이며, 확률은 {prob:.2f}입니다.
    #     다양한 설명 알고리즘({', '.join(explanation.index)})을 사용하여 해당 예측에 주요한 영향을 미치는 특징을 설명합니다.
    #     여러 설명 기법에서 공통적으로 나타나는 중요한 feature는 다음과 같습니다:
    #     {', '.join(explanation.columns)}
    #     """


def launch_app(xai_app: XAIApp):
    with gr.Blocks(
        theme=xai_app.theme, 
        # css=xai_app.css, 
        # js=xai_app.js, 
        # head=xai_app.head, 
        title="[Finance] Plug and Play") as demo:
        gr.set_static_paths("./")
        gr.HTML(xai_app.show_title())

        
        with gr.Tab("프로젝트 개요"):
            gr.HTML(xai_app.show_project_description())

        with gr.Tab("데이터 및 모델 정보"):
            gr.HTML(xai_app.show_data_model_info())
        
        with gr.Tab("전역 설명(Global explanation)"):
            gr.HTML(xai_app.show_global_explanation())
            for model_name in xai_app.trained_models.keys():
                def show_global_explanation_plot(model_name):
                    fig = xai_app.load_global_explanation(model_name)
                    if fig is None:
                        example = gr.State("example")
                        labels = gr.Label(f"아직 {model_name} 모델에 대해 전역 설명이 생성되지 않았습니다.")
                        bttn = gr.Button(f"{model_name} 모델에 대한 전역 설명 생성")
                        output = gr.Textbox(label="전역 설명 생성 결과")
                        desc = gr.HTML(visible=True)
                        plot_output = gr.Plot(label=f"{model_name} 모델에 대한 전역 설명", visible=True)

                        def generate_global_explanation():
                            response = xai_app.create_global_explanation(model_name)
                            return response

                        def handle_click():
                            desc = None
                            generate_global_explanation()
                            new_fig = xai_app.load_global_explanation(model_name)
                            if new_fig:
                                desc = xai_app.show_global_explanation(model_name)

                            return [f"{model_name} 모델에 대한 전역 설명이 생성되었습니다.", desc, new_fig]

                        bttn.click(handle_click, outputs=[output, desc, plot_output])
                    else:
                        gr.HTML(xai_app.show_global_explanation(model_name))
                        gr.Plot(fig)

                show_global_explanation_plot(model_name)

        with gr.Tab("개별 데이터 설명(Local explanation)"):
            gr.HTML("<h1>데이터 조회</h1>")
            
             # Search input and DataFrame display
            search_input = gr.Textbox(label=f"인덱스로 검색하기(최대 : {xai_app.preprocessed_data.shape[0] - 1})")
            database = xai_app.raw_data.reset_index(drop=True).reset_index(drop=False)
            df = gr.DataFrame(value=database)
            bttn = gr.Button("데이터 조회하기")

            def filter_dataframe(search_text):
                if search_text == "":
                    filtered_df = database
                else:
                    try:
                        index = int(search_text)
                        if 0 <= index < len(database):
                            filtered_df = database.loc[[index]]
                        else:
                            filtered_df = pd.DataFrame(columns=database.columns)  # Empty DataFrame
                    except ValueError:
                        filtered_df = pd.DataFrame(columns=database.columns)  # Empty DataFrame
                return filtered_df

            bttn.click(filter_dataframe, inputs=search_input, outputs=df)

            for model_name in xai_app.trained_models.keys():
                def create_model_ui(model_name):
                    gr.HTML(f"<h1>모델: {model_name}</h1>")

                    rec_explainers = xai_app.recommend_explainers(xai_app.detect_model_type(model_name))
                    gr.HTML(f"""<h3> 설명 알고리즘 선택하기 </h3> <p>해당 모델에 대해 적용가능한 설명 알고리즘만 추천합니다.</p>""")
                    explainer_checkbox = gr.CheckboxGroup(choices=rec_explainers, value=rec_explainers, label="적용 가능한 설명 알고리즘")
                    index_input = gr.Number(label=f"데이터 번호(최대 : {xai_app.preprocessed_data.shape[0] - 1})")
                    explain_button = gr.Button("설명 생성하기")

                    with gr.Row():
                        prediction_result = gr.Label(label="모델 예측 결과")

                    plots = []
                    n_rows = len(rec_explainers) // 2
                    for row in range(n_rows):
                        with gr.Row():
                            for i in range(2):
                                plots.append(gr.Plot(label=f"설명 결과 {row*2 + i + 1}"))

                    def generate_explanation_and_output(index, explainers):
                        prediction_result_text = xai_app.get_prediction_text(model_name, index)
                        explanation_figs = xai_app.generate_explanation(model_name, index, explainers)
                        plot_values = list(explanation_figs.values())
                        # Fill the rest of the plot outputs with None if fewer than 4
                        while len(plot_values) < len(rec_explainers):
                            plot_values.append(None)
                        return [prediction_result_text] + plot_values

                    explain_button.click(
                        generate_explanation_and_output,
                        inputs=[index_input, explainer_checkbox],
                        outputs=[prediction_result] + plots
                    )
                
                create_model_ui(model_name)
        
            
        with gr.Tab("설명에 대한 평가 지표"):
            with gr.Accordion("평가 지표들에 대한 설명 보기", open=False):
                gr.HTML(xai_app.show_evaluation_metric_decription())
            for model_name in xai_app.trained_models.keys():
                if model_name != "TabResNet":
                    continue

                gr.HTML(f"<h1>모델: {model_name}</h1>")
                def create_eval_ui(model_name):
                    gr.HTML(f"<h2> {model_name} 모델에 대한 평가 지표 </h2>")
                    eval_metrics = gr.CheckboxGroup(choices=xai_app.eval_metrics, value=xai_app.eval_metrics, label="평가 지표 선택하기")
                    eval_button = gr.Button("평가 지표 보기")
                    metrics_outputs = []
                    for metric_name in xai_app.eval_metrics:
                        metrics_outputs.append(gr.Plot(label=f"{model_name}_{metric_name} 평가 지표"))

                    def plot_eval_metrics(eval_metrics):
                        outputs = []
                        for eval_metric in eval_metrics:
                            metric_plot = xai_app.show_evaluation_metrics(model_name, eval_metric)
                            outputs.append(metric_plot)
                        
                        while len(outputs) < len(xai_app.eval_metrics):
                            outputs.append(None)
                        return outputs

                    eval_button.click(
                        plot_eval_metrics,
                        inputs=eval_metrics,
                        outputs=metrics_outputs
                    )

                create_eval_ui(model_name)
    
    return demo


# if __name__ == "__main__":
raw_data = pd.read_csv("data/baf/preprocess/test.csv", index_col=0)
with open("data/baf/preprocess/X_test.npy", 'rb') as f:
    preprocessed_data = np.load(f)

with open("data/baf/preprocess/X_train.npy", 'rb') as f:
    X_train = np.load(f)


with open("data/baf/preprocess/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

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

trained_models = {
    "Logistic Regression": model_lr,
    "Random Forest": model_rf,
    "XGBoost": model_xgb,
    "TabResNet": model_nn,
}

xai_app = XAIApp(raw_data.drop(columns=['fraud_bool']), preprocessed_data, metadata, trained_models, bg_data=X_train)
demo = launch_app(xai_app)
demo.launch(favicon_path="static/XAI-Top-PnP.svg")
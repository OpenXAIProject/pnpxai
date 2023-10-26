# Model Detector

* 추출할 정보
  * torch.nn.Module인 경우만 추출
  * Explainer에 필요한 정보
    * 특정 레이어(CNN, Self attention)를 보유하고 있는지 없는지
    * 특정 레이어를 보유하고 있다면 그 위치는 어디인지
    * 기타 정보
      * Residual Connection을 사용하는 모델인지 아닌지
    * Visualizer에 필요한 정보
      * 모든 개별 레이어의 이름에 대한 정보

\-----------------------------------------------------------------------------------------

* Required Information
  * Only for `torch.nn.Module` type model
  * Information Needed for the Explainer:
    * Whether it contains specific layers (like CNN, Self-attention).
    * The location of these specific layers if they are present.
    * Whether the model uses Residual Connections.
  * Information Needed for the Visualizer:
    * Information about the names of each individual layer.

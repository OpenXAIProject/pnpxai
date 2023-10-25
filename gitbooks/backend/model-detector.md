# Model Detector

* torch.nn.Module로 주어진 모든 모델에 대해서 필요한 데이터를 추출
  * Explainer에 필요한 정보
    * 특정 레이어(CNN, Self attention)를 보유하고 있는지 없는지
    * 특정 레이어를 보유하고 있다면 그 위치는 어디인지
    * 기타 정보
      * Residual Connection을 사용하는 모델인지 아닌지
  * Visualizer에 필요한 정보
    * 모든 개별 레이어의 이름에 대한 정보

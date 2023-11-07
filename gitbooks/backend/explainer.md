# Explainer

* Algorithm Coverage : 아래의 알고리즘들을 시각화하는 것이 목표
  * Integrated Gradient
  * LIME
  * LRP
  * GradCAM
  * Attention Map
  * Saliency Map
  * RAP
* 특이사항
  * Residual Connection이 있는 모델의 경우 LRP 알고리즘을 적용하는데 문제가 생길 수 있으므로 Residual connection이 있는 경우 LRP 삭제

\----------------------------------------------------------------------------------------------------

* The goal is to visualize the following algorithms:
  * Integrated Gradient
  * LIME
  * LRP
  * GradCAM
  * Attention Map
  * Saliency Map
  * RAP

## Special Note

* For models with Residual Connections, there may be issues applying the LRP algorithm. Therefore, remove LRP in cases where Residual Connections are present.

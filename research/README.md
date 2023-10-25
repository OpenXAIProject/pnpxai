---
description: 모든 개발자들이 공통으로 참조할 사항
---

# Commons

### User flow

1. .py 또는 .ipynb 환경에서 모델과 sample 데이터를 불러온다.
2. PnP XAI 패키지의 object를 생성한다.
3. 모델, 샘플 데이터, 해당 모델에 관한 question을 object의 explain이라는 method의 argument로 넘긴다.
4. launch라는 method를 사용해서 Web application을 실행한다.
   1. Model을 자동으로 탐지하고, 해당 모델의 구조를 요약해서 시각화할 수 있게 한다.
   2. Model과 Sample 데이터에 대해 XAI 알고리즘을 적용한 결과를 시각화한다.
   3. 적용한 XAI 알고리즘에 점수를 매긴다.

### Technical Base

* Backend
  * Python(torch, Flask)
* UI Design
  * Figma
* Frontend
  * React.js
* Collaborate
  * Youtrack
* Documentation
  * Product Documentation : Notion
  * API Documentation : Sphinx

### Common Configuration

#### UI Type

code를 기초로 model, sample 데이터를 argument로 받는 object를 만들고, 이 object를 기반으로 Web Application을 띄우는 방식

#### Coverage

* Model Coverage
  * Image Classification Model을 대상으로 한다.
  * 그 중 torch.nn.Module을 상속한 타입의 모델만을 대상으로 한다.

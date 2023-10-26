---
description: "\bCommon information for all developers"
---

# Commons

## User flow

1. .py 또는 .ipynb 환경에서 모델과 sample 데이터를 불러온다.
2. PnP XAI 패키지의 object를 생성한다.
3. 모델, 샘플 데이터, 해당 모델에 관한 question을 object의 explain이라는 method의 argument로 넘긴다.
4. launch라는 method를 사용해서 Web application을 실행한다.
   1. Model을 자동으로 탐지하고, 해당 모델의 구조를 요약해서 시각화할 수 있게 한다.
   2. Model과 Sample 데이터에 대해 XAI 알고리즘을 적용한 결과를 시각화한다.
   3. 적용한 XAI 알고리즘에 점수를 매긴다.
   4. XAI 알고리즘에 대한 설명을 볼 수 있는 페이지를 만든다.

## Technical Base

* Backend
  * Python(torch, Flask)
* UI Design
  * Figma
* Frontend
  * React.js
* Collaborate
  * Git Project, Git Book
* Documentation
  * Product Documentation : Notion, Gitbook
  * API Documentation : Sphinx

## Common Setting

#### UI Type

2가지 방식의 UI를 모두 구현해야 한다.

Code Base UI

Model, sample 데이터를 argument로 받는 object를 만들어서 기본적인 기능을 실행함.

Web Base UI

코드에서 서버를 launch할 수 있고, 이를 실행하면 local 환경에서 Web Application이 실행됨.

#### Coverage

* Image Classification Task를 수행하는 모델을 대상으로 한다.
* 그 중 torch.nn.Module을 상속한 타입의 모델만을 대상으로 한다.
* Algorithm Coverage : 아래의 알고리즘들을 시각화하는 것이 목표
  * Integrated Gradient
  * LIME
  * LRP
  * GradCAM
  * Attention Map
  * Saliency Map
  * RAP

\-------------------------------------------------------------------------------------------

### User Flow

1. Load the model and sample data in a `.py` or `.ipynb` environment.
2. Create an object of the PnP XAI package.
3. Pass the model, sample data, and questions about the model as arguments to the `explain` method of the object.
4. Use the `launch` method to run the Web application.
5. Automatically detect the model and enable visualization by summarizing the model's structure.
6. Visualize the results of applying XAI algorithms to the model and sample data.
7. Score the applied XAI algorithms.
8. Create a page to view explanations about the XAI algorithms.

### Technical Base

#### Backend

* Python (torch, Flask)

#### UI Design

* Figma

#### Frontend

* React.js

#### Collaborate

* Git Project, Git Book

#### Documentation

* Product Documentation: Notion
* API Documentation: Sphinx

### Common Setting

#### UI Type

* Implement both types of UIs.
  * **Code Base UI**: Create an object that receives the model and sample data as arguments and executes basic functions.
  * **Web Base UI**: Able to launch the server from the code, which then runs the Web Application in a local environment.



**Model Scope**

* Focus on models performing the Image Classification Task.
* Specifically, target models that are a type of `torch.nn.Module`.

#### XAI Algorithm Scope

* The goal is to visualize the following algorithms:
  * Integrated Gradient
  * LIME
  * LRP
  * GradCAM
  * Attention Map
  * Saliency Map
  * RAP

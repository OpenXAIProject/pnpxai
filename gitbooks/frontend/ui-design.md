# UI Design

## Figma Link

[https://www.figma.com/file/PXovLiXK5fB3zferETadhu/Web-Design-File?type=design\&node-id=0-1\&mode=design\&t=tA7X26k0vT5LUZGg-0](https://www.figma.com/file/PXovLiXK5fB3zferETadhu/Web-Design-File?type=design\&node-id=0-1\&mode=design\&t=tA7X26k0vT5LUZGg-0)

## Design 요구사항

* 주요 페이지 구성
  * Model Detection Info
    * Model Name
    * Model Layer(기본적으로 앞의 라인 중 일부만 보여주고, 토글을 누르면 마지막 5줄 포함 최대 30줄까지 출력)
  * Algorithm Visualization
    * 주어진 Sample들에 대해서 아래의 정보를 보여줌
    * Sample Image, Predicted Label, True Label, Overlapped Heatmap, Evaluation Score
    * 알고리즘을 선택할 수 있도록
  * Explanation for algorithm
    * 페이지 설명 : 전반적인 알고리즘 설명서
    * figure 설명 : Figure에 hover를 했을 때 볼 수 있도록

\-----------------------------------------------------------------------------------------------------

## Design Requirements

### Main Page Composition

* **Model Detection Info**
  * Model Name
  * Model Layer: Initially display only a portion of the earlier lines. When a toggle is clicked, display up to the last 5 lines, with a maximum of 30 lines.
* **Algorithm Visualization**
  * Show the following information for the given samples:
    * Sample Image
    * Predicted Label
    * True Label
    * Overlapped Heatmap
    * Evaluation Score
  * Enable selection of algorithms

### Explanation for Algorithm

* **Page Description**: A comprehensive guide to the overall algorithm.
* **Figure Description**: Allow visibility when hovering over a figure.

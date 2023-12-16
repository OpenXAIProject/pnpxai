import { ExperimentResult } from "../app/types";

const modifyLayout = (layout: any) => {
  const modifiedLayout = {
      ...layout,
      xaxis: { visible: false },
      yaxis: { visible: false },
      width: 240,
      height: 200,
      margin: { l: 0, r: 0, t: 0, b: 0 }
  };

  modifiedLayout.template.data.heatmap[0].colorbar = null;

  return modifiedLayout;
}

const modifyData = (data: any) => {
    const modifiedData = {
      ...data[0],
      coloraxis: null,
      showscale: false,
    }

    return [modifiedData];
}

export const preprocess = (response: any) => {
    response.data.data.forEach((result: any) => {
        result.input = JSON.parse(result.input);
        result.input.layout = modifyLayout(result.input.layout);

        result.explanations.forEach((explanation: any) => {
          if (explanation.data !== null) {
            explanation.data = JSON.parse(explanation.data);
            explanation.data.data = modifyData(explanation.data.data);
            explanation.data.layout = modifyLayout(explanation.data.layout);
          }
        });
    });
}


export const AddMockData = (response: any) => {
  response.data.data.forEach((result: any) => {
    result.prediction = {
      "label": "cat",
      "isCorrect": true,
      "probPredictions": [
        {
          "label": "cat",
          "score": 90.5
        },
        {
          "label": "dog",
          "score": 5.3
        },
        {
          "label": "fish",
          "score": 4.2
        }
      ]
    }
  })
}
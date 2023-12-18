const colorScale = [
  [
      0,
      "rgb(255,245,240)"
  ],
  [
      0.125,
      "rgb(254,224,210)"
  ],
  [
      0.25,
      "rgb(252,187,161)"
  ],
  [
      0.375,
      "rgb(252,146,114)"
  ],
  [
      0.5,
      "rgb(251,106,74)"
  ],
  [
      0.625,
      "rgb(239,59,44)"
  ],
  [
      0.75,
      "rgb(203,24,29)"
  ],
  [
      0.875,
      "rgb(165,15,21)"
  ],
  [
      1,
      "rgb(103,0,13)"
  ]
];

const modifyLayout = (layout: any) => {
  const modifiedLayout = {
      ...layout,
      template: null,
      coloraxis: {colorscale : colorScale, showscale: false},
      xaxis: { visible: false },
      yaxis: { visible: false },
      width: 240,
      height: 200,
      margin: { l: 0, r: 0, t: 0, b: 0 },
  };


  return modifiedLayout;
}

const modifyData = (data: any) => {
    const modifiedData = {
      ...data[0], 
      z : data[0].z.slice().reverse(),
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

  return response;
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
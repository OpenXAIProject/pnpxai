// File: utils.tsx
import ColorScales from '../assets/styles/colorScale.json';

interface ColorScales {
  [key: string]: { [key: string]: any[][] };
}

const colorScales: ColorScales = ColorScales;

const modifyLayout = (layout: any, params: any) => {
  // TODO : Get colorType from params (Backend)
  let colorType = 'seq';
  let colorKey = params.colorScale[colorType];
  const modifiedLayout = {
      ...layout,
      template: null,
      coloraxis: {colorscale : colorScales[colorType][colorKey], showscale: false},
      xaxis: { visible: false },
      yaxis: { visible: false },
      width: 200,
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

export const preprocess = (response: any, params: any) => {
  response.data.data.forEach((result: any) => {
    result.input = JSON.parse(result.input);
    result.input.layout = modifyLayout(result.input.layout, params);

    result.explanations.forEach((explanation: any) => {
      if (explanation.data !== null) {
        explanation.data = JSON.parse(explanation.data);
        explanation.data.data = modifyData(explanation.data.data);
        explanation.data.layout = modifyLayout(explanation.data.layout, params);
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
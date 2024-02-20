import { Metric, Nickname, ColorScales, HelpText, ExperimentResult } from '../app/types';
import ColorScalesData from '../assets/styles/colorScale.json';

const colorScales: ColorScales = ColorScalesData;

export const nickname: Nickname[] = [
    { "name": "MuFidelity", "nickname": "Correctness" },
    { "name": "Sensitivity", "nickname": "Continuity" },
    { "name": "Complexity", "nickname": "Compactness" },
  ];
  
export const explainerNickname: Nickname[] = [
  { "name": "GuidedGradCam", "nickname": "Guided Grad-CAM" },
  { "name": "IntegratedGradients", "nickname": "Integrated Gradients" },
  { "name": "KernelShap", "nickname": "kernelSHAP" },
  { "name": "LRP", "nickname": "LRP" },
  { "name": "Lime", "nickname": "LIME" },
  { "name": "RAP", "nickname": "RAP" },
];

export const domain_extension_plan = "The task will be extended to other domains(Tabular, Text, Time Series) in the future."

export const helptext: HelpText = {
  "Correctness" : "the truthfulness/reliability of explanations about a prediction model (AI model). That is, it indicates how truthful the explanation is compared to the operation of the black box model.",
  "Continuity" : "how continuous (i.e., smooth) an explanation is. An explanation function with high continuity ensures that small changes in the input do not bring about significant changes in the explanation.",
  "Compactness" : "the size/amount of an explanation. It ensures that complex and redundant explanations that are difficult to understand are not presented.",
  // "Completeness" : " the extent to which a prediction model (AI model) is explained. Providing 'the whole truth' of the black box model represents high completeness, but a good explanation should balance conciseness and correctness.",
}

export const routes = [
  {
    path: "/model-info",
    name: "Experiment Information"
  },
  {
    path: "/model-explanation",
    name: "Local Explanation"
  }
];


export const metricSortOrder = new Map<string, number>(nickname.map((item, index) => [item.name, index]));

export  function sortMetrics(metrics: Metric[], sortOrder: Map<string, number>): Metric[] {
  return metrics.sort((a, b) => {
    let orderA = sortOrder.get(a.name) ?? Number.MAX_VALUE;
    let orderB = sortOrder.get(b.name) ?? Number.MAX_VALUE;
    return orderA - orderB;
  });
}


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


export const changeColorMap = (experimentResults: ExperimentResult[], params: any) => {
  experimentResults.forEach((result: ExperimentResult) => {
    result.explanations.forEach((explanation: any) => {
      if (explanation.data !== null) {
        explanation.data.layout = modifyLayout(explanation.data.layout, params);
      }
    });
  });

  return experimentResults;
}
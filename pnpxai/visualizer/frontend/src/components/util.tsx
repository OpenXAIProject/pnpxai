import { Metric, Nickname } from '../app/types';

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
  
  
  export const metricSortOrder = new Map<string, number>(nickname.map((item, index) => [item.name, index]));
  
  export  function sortMetrics(metrics: Metric[], sortOrder: Map<string, number>): Metric[] {
    return metrics.sort((a, b) => {
      let orderA = sortOrder.get(a.name) ?? Number.MAX_VALUE;
      let orderB = sortOrder.get(b.name) ?? Number.MAX_VALUE;
      return orderA - orderB;
    });
  }
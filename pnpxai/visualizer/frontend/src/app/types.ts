// src/app/types.ts
interface Project {
    id: string;
    experiments: Experiment[];  // Specify the correct type
  }
  
// Define a type for Experiment as well
interface Experiment {
  id: string;
  name: string;
  inputs: InputData[];  // Specify the correct type
  model: Model;
  explainers: Explainer[];
  metrics: Metric[];
  modelDetected: boolean;
}

interface Explainer {
  id: number;
  name: string;
}

interface Metric {
  id: number;
  name: string;
}

interface Model {
  id: number;
  name: string;
  nodes: any[];
  edges: any[];
}

// Define input data type
interface InputData {
  id: string;
  source: string;
}

interface ExperimentResult {
  input : {
    data: [{
      name: string;
    }]
    layout: {};
  };
  target: string;
  outputs: {
    key: string;
    value: number;
  }[];
  explanations: {
    explainer : string;
    data: {
      data : [{
        name: string;
        z: number[][];
      }]
      layout: {};
    };
    evaluation: {
      MuFidelity: number;
      Sensitivity: number;
      Complexity: number;
    };
    rank : number;
  }[];
}

interface Nickname {
  name: string;
  nickname: string;
}

interface Config {
  colorMap: ColorMap;
}
interface ColorMap {
  seq : string;
  diverge : string;
}

interface Status {
  currentProject: string;
  currentExp: string;
}

interface Cache {
  // For each project and experiment, cache holds the data
  [key: string]: {
    [key: string]: ExCache;
  };
}
interface ExCache {
  inputs: InputData[];
  explainers: Explainer[];
  metrics: Metric[];
}

export type { 
  Project, Experiment, Explainer, Metric, Model, 
  InputData, ExperimentResult, Nickname,
  Config, Status, Cache
}

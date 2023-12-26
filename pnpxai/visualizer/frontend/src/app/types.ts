// src/app/types.ts
interface Project {
    id: string;
    experiments: Experiment[];  // Specify the correct type
  }
  
// Define a type for Experiment as well
interface Experiment {
  id?: string;
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
  id: number;
  imageObj: imageObj;
}

interface imageObj {
  data: any;
  layout: any;
}

interface ExperimentResult {
  input : {
    data: [{
      name: string;
    }]
    layout: {};
  };
  target: string;
  outputs: string[];
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

export type { Project, Experiment, Explainer, Metric, Model, InputData, imageObj, ExperimentResult}

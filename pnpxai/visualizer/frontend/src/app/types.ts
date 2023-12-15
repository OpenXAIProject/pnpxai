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
  modelDetected: boolean;
}

interface Explainer {
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
  visualizations: {
    explainer : string;
    data: {
      data : [{
        name: string;
        z: number[][];
      }]
      layout: {};
    };
    metrics: {
      faithfulness: number;
      robustness: number;
    };
  }[];
  prediction: {
    label: string;
    probPredictions: {
      label: string;
      score: number;
    }[];
    isCorrect: boolean;
  };
}

export type { Project, Experiment, Explainer, Model, InputData, imageObj, ExperimentResult}

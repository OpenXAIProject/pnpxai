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
  id: string;
  imageObj: imageObj;
}

interface imageObj {
  data: any;
  layout: any;
}

export type { Project, Experiment, Explainer, Model, InputData, imageObj}

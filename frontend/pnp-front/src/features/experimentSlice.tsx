import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import plotly_test from '../assets/mockup/plotly_test.json'

// Define a type for the slice state
interface Sample {
  "sample_id" : number;
  "name" : string;
  "json" : string; // Plotly Image JSON
}

interface Experiment {
  "experiment_id": number;
  "name": string;
  "model" : string;
  "modelDetected": boolean;
  "modelStructure": string;
  "algorithms": string[];
  "data" : Sample[];
}

interface ExperimentState {
  data: Experiment[];
}

// Initial state
const initialState: ExperimentState = {
  data: plotly_test,
};

export const experimentSlice = createSlice({
  name: 'experiments',
  initialState,
  reducers: {
    setData: (state, action: PayloadAction<any>) => {
      state.data = action.payload;
    },
  },
});

// Actions
export const { setData } = experimentSlice.actions;

// Asynchronous thunk action
export const fetchData = () => async (dispatch: any) => {
  // Fetch data from the server and dispatch setData
  // ...
};

export default experimentSlice.reducer;

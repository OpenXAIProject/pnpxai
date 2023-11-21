// src/features/explainSlice.tsx
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

// Define a type for the slice state
interface ExplainState {
  selectedData: number[];
  selectedAlgorithms: number[];
  labelData: { id: number; realLabel: string; predictedLabel: string; evaluationResults: any }[];
  plotData: any[];
}

// Define the initial state using that type
const initialState: ExplainState = {
  selectedData: [],
  selectedAlgorithms: [],
  labelData: [
    //... your mock data here
  ],
  plotData: [],
};

export const explainSlice = createSlice({
  name: 'explain',
  initialState,
  reducers: {
    setSelectedData: (state, action: PayloadAction<number[]>) => {
      state.selectedData = action.payload;
    },
    setSelectedAlgorithms: (state, action: PayloadAction<number[]>) => {
      state.selectedAlgorithms = action.payload;
    },
    setLabelData: (state, action: PayloadAction<{ id: number; realLabel: string; predictedLabel: string; evaluationResults: any }[]>) => {
      state.labelData = action.payload;
    },
    setPlotData: (state, action: PayloadAction<any[]>) => {
      state.plotData = action.payload;
    },
    // Add more reducers as needed
  },
});

// Action creators are generated for each case reducer function
export const { setSelectedData, setSelectedAlgorithms, setLabelData, setPlotData } = explainSlice.actions;

export default explainSlice.reducer;

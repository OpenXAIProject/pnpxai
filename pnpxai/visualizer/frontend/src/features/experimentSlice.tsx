// // src/features/experimentSlice.tsx
// import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
// import axios from 'axios';
// import { Experiment } from '../app/types';
// import { fetchInputsByExperimentId as fetchExperimentsApi } from './apiService';

// interface ExperimentState {
//   experimentsByProjectId: { [key: string]: Experiment[] };
// }

// const initialState: ExperimentState = {
//   experimentsByProjectId: {},
// };

// // Revised async thunk for fetching experiments
// export const fetchExperimentsByProjectId = createAsyncThunk(
//   'experiments/fetchByProjectId',
//   async (projectId: string, { rejectWithValue }) => {
//     try {
//       const response = await fetchExperimentsApi(projectId);
//       return { projectId, experiments: response.data.data as Experiment[] };
//     } catch (error : any) {
//       return rejectWithValue(error.response.data);
//     }
//   }
// );

// const experimentSlice = createSlice({
//   name: 'experiments',
//   initialState,
//   reducers: {
//     // Reducers here if necessary
//   },
//   extraReducers: (builder) => {
//     builder.addCase(fetchExperimentsByProjectId.fulfilled, (state, action) => {
//       const { projectId, experiments } = action.payload;
//       state.experimentsByProjectId[projectId] = experiments;
//     });
//     // Handle other cases like pending, rejected if needed
//   },
// });

// export default experimentSlice.reducer;

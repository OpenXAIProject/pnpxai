import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { fetchProjects as fetchProjectsApi, fetchModelsByProjectId, fetchInputsByExperimentId } from './apiService';
import { Project, Experiment, Model, InputData, imageObj } from '../app/types';
import { Input } from '@mui/material';

const initialState = {
  data: [] as Project[], // Initialize as an empty array
  loaded: false, // Add a flag to track if the data is loaded
};

export const fetchProjects = createAsyncThunk(
  'projects/fetchProjects',
  async (_, { rejectWithValue }) => {
    try {
      const projectsResponse = await fetchProjectsApi();
      let projects = projectsResponse.data.data as Project[];

      for (const project of projects) {
        try {
          const modelsResponse = await fetchModelsByProjectId(project.id);
          const models = modelsResponse.data.data as Model[];

          for (let i = 0; i < project.experiments.length; i++) {
            const experiment = project.experiments[i];
            experiment.id = experiment.name;
            experiment.model = models[i];
            experiment.modelDetected = true;

            try {
              const inputsResponse = await fetchInputsByExperimentId(project.id, experiment.id);
              experiment.inputs = inputsResponse.data.data.map((input: string, index: number) => {
                const parsedInput = JSON.parse(input);
              
                return {
                  id: `${index}`,
                  imageObj: {
                    data: parsedInput.data,
                    layout: parsedInput.layout,
                  } as imageObj,
                };
              });
            } catch (inputsError) {
              console.error('Error fetching inputs:', inputsError);
              // Handle or ignore input errors
            }
          }
        } catch (modelsError) {
          console.error('Error fetching models:', modelsError);
          // Handle or ignore model errors
        }
      }
      return projects;
    } catch (err: any) {
      return rejectWithValue(err.response.data);
    }
  }
);

const projectSlice = createSlice({
  name: 'projects',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder.addCase(fetchProjects.fulfilled, (state, action) => {
      state.data = action.payload;
      state.loaded = true; // Set the flag when data is loaded
    });
    // Handle other cases
  },
});

export default projectSlice.reducer;



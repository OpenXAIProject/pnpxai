import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { fetchProjects as fetchProjectsApi, fetchModelsByProjectId } from './apiService';
import { Project, Model } from '../app/types';
import { Metric } from '../app/types';

const initialState = {
  data: [] as Project[], // Initialize as an empty array
  currentProject: {} as Project,
  loaded: false, // Add a flag to track if the data is loaded
  error: false,
  colorMap: {'seq' : 'Reds', 'diverge' : 'bwr'}
};

// TODO: change this nickname to the real name
interface Nickname {
  name: string;
  nickname: string;
}

const nickname: Nickname[] = [
  { "name": "MuFidelity", "nickname": "Correctness" },
  { "name": "Sensitivity", "nickname": "Continuity" },
  { "name": "Complexity", "nickname": "Compactness" },
];

const explainerNickname: Nickname[] = [
  { "name": "GuidedGradCam", "nickname": "Guided Grad-CAM" },
  { "name": "IntegratedGradients", "nickname": "Integrated Gradients" },
  { "name": "KernelShap", "nickname": "kernelSHAP" },
  { "name": "LRP", "nickname": "LRP" },
  { "name": "Lime", "nickname": "LIME" },
  { "name": "RAP", "nickname": "RAP" },
];


const metricSortOrder = new Map<string, number>(nickname.map((item, index) => [item.name, index]));

function sortMetrics(metrics: Metric[], sortOrder: Map<string, number>): Metric[] {
  return metrics.sort((a, b) => {
    let orderA = sortOrder.get(a.name) ?? Number.MAX_VALUE;
    let orderB = sortOrder.get(b.name) ?? Number.MAX_VALUE;
    return orderA - orderB;
  });
}


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
            
            if (experiment.metrics) {
              experiment.metrics = sortMetrics(experiment.metrics, metricSortOrder);
            }
            
            for (let j = 0; j < experiment.explainers.length; j++) {
              const explainer = experiment.explainers[j];
              explainer.name = explainerNickname.find((item) => item.name === explainer.name)?.nickname ?? explainer.name;
            }

            experiment.model = models[i];
            experiment.modelDetected = true;

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
  reducers: {
    // Define setCurrentProject reducer
    setCurrentProject(state, action: PayloadAction<string>) {
      const projectId = action.payload;
      const foundProject = state.data.find(project => project.id === projectId);
      if (foundProject) {
        state.currentProject = foundProject;
      }
    },
    setColorMap(state, action: PayloadAction<any>) {
      state.colorMap = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
    .addCase(fetchProjects.pending, (state) => {
      state.error = false; // Reset error state on new fetch
    })
    .addCase(fetchProjects.fulfilled, (state, action) => {
      state.data = action.payload;
      state.currentProject = action.payload[0] || {} as Project;
      state.loaded = true;
      state.error = false; // Reset error state on successful fetch
    })
    .addCase(fetchProjects.rejected, (state) => {
      state.error = true; // Set error state on fetch failure
    });
  },
});
export const { setCurrentProject, setColorMap } = projectSlice.actions;

export default projectSlice.reducer;



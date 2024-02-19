import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { fetchProjects as fetchProjectsApi, fetchModelsByProjectId } from './apiService';
import { Project, Model, ExperimentCache, Status } from '../app/types';
import { sortMetrics, metricSortOrder, explainerNickname } from '../components/util';

const initialState = {
  loaded: false,
  error: false,
  projects: [] as Project[],
  status: {} as Status,
  cache: {} as ExperimentCache[]
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
      const foundProject = state.projects.find(project => project.id === projectId);
      if (foundProject) {
        state.status.currentProject = foundProject.id;
      }
    },
    setColorMap(state, action: PayloadAction<{ projectId: string; expId: string; colorMap: any }>) {
      const { projectId, expId, colorMap } = action.payload;
      const foundCache = state.cache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.config.colorMap = colorMap;
      }
    }
  },
  extraReducers: (builder) => {
    builder
    .addCase(fetchProjects.pending, (state) => {
      state.error = false; // Reset error state on new fetch
    })
    .addCase(fetchProjects.fulfilled, (state, action) => {
      state.projects = action.payload;
      state.status.currentProject = action.payload[0].id;
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



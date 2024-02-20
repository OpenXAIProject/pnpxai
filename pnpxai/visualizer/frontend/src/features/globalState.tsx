import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { fetchProjects as fetchProjectsApi, fetchModelsByProjectId } from './apiService';
import { Project, Model, ProjectCache, ExperimentCache, Status, ExperimentResult } from '../app/types';
import { sortMetrics, metricSortOrder, explainerNickname } from '../components/util';
import colorScales from '../assets/styles/colorScale.json';

const initialState = {
  loaded: false,
  error: false,
  projects: [] as Project[],
  status: {} as Status,
  projectCache: [] as ProjectCache[],
  expCache: [] as ExperimentCache[]
};

export const fetchProjects = createAsyncThunk(
  'projects/fetchProjects',
  async (_, { dispatch, rejectWithValue }) => {
    try {
      const projectsResponse = await fetchProjectsApi();
      let projects = projectsResponse.data.data as Project[];
      let projectCacheUpdates: ProjectCache[] = [];
      let expCacheUpdates: ExperimentCache[] = [];

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

            // Set Default Config
            const defaultExplainers = experiment.explainers
            const defaultMetrics = experiment.metrics.filter(metric => metric.name === 'MuFidelity');
            const defaultColorMap = {
              seq : Object.keys(colorScales.seq)[0], 
              diverge : Object.keys(colorScales.diverge)[0]
            };

            expCacheUpdates.push({
              projectId: project.id,
              expId: experiment.id,
              galleryInputs: [],
              inputs: [],
              explainers: defaultExplainers,
              metrics: defaultMetrics,
              experimentResults: [],
              config: {}
            });

            projectCacheUpdates.push({
              projectId: project.id,
              config: {
                colorMap: defaultColorMap
              }
            });

          }
        } catch (modelsError) {
          console.error('Error fetching models:', modelsError);
          // Handle or ignore model errors
        }
      }

      dispatch(globalState.actions.setExpCache(expCacheUpdates));
      dispatch(globalState.actions.setProjectCache(projectCacheUpdates));

      return projects;
    } catch (err: any) {
      return rejectWithValue(err.response.data);
    }
  }
);

const globalState = createSlice({
  name: 'globalState',
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
    setProjectCache(state, action: PayloadAction<ProjectCache[]>) {
      state.projectCache = action.payload;
    },
    setExpCache(state, action: PayloadAction<ExperimentCache[]>) {
      state.expCache = action.payload;
    },
    setGalleryInputs(state, action: PayloadAction<{ projectId: string; expId: string; galleryInputs: any }>) {
      const { projectId, expId, galleryInputs } = action.payload;
      const foundCache = state.expCache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.galleryInputs = galleryInputs;
      }
    },
    setInputs(state, action: PayloadAction<{ projectId: string; expId: string; inputs: any }>) {
      const { projectId, expId, inputs } = action.payload;
      const foundCache = state.expCache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.inputs = inputs;
      }
    },
    setExplainers(state, action: PayloadAction<{ projectId: string; expId: string; explainers: any }>) {
      const { projectId, expId, explainers } = action.payload;
      const foundCache = state.expCache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.explainers = explainers;
      }
    },
    setMetrics(state, action: PayloadAction<{ projectId: string; expId: string; metrics: any }>) {
      const { projectId, expId, metrics } = action.payload;
      const foundCache = state.expCache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.metrics = metrics;
      }
    },
    setExperimentResults(state, action: PayloadAction<{ projectId: string; expId: string; experimentResults: ExperimentResult[] }>) {
      const { projectId, expId, experimentResults } = action.payload;
      const foundCache = state.expCache.find(cache => cache.projectId === projectId && cache.expId === expId);
      if (foundCache) {
        foundCache.experimentResults = experimentResults;
      }
    },
    setColorMap(state, action: PayloadAction<{ projectId: string; colorMap: any }>) {
      const { projectId, colorMap } = action.payload;
      const foundCache = state.projectCache.find(cache => cache.projectId === projectId);
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
export const { 
  setCurrentProject,
  setGalleryInputs,
  setInputs,
  setExplainers,
  setMetrics,
  setExperimentResults,
  setColorMap 
} = globalState.actions;

export default globalState.reducer;



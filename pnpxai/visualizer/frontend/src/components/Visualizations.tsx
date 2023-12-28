// src/components/Visualizations.tsx
import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { Typography, Box, ImageList, ImageListItem, CircularProgress, Grid, Divider} from '@mui/material';
import Plot from 'react-plotly.js';
import { ExperimentResult } from '../app/types';
import { RunExperiment, fetchExperiment } from '../features/apiService';
import { preprocess, AddMockData } from './utils';


const Visualizations: React.FC<{ 
  experiment: string; inputs: number[]; explainers: number[]; metrics: number[]; loading: boolean; setLoading: any
}> = ({ experiment, inputs, explainers, metrics, loading, setLoading }) => {
  const nickname = [
    {"name": "Complexity", "nickname": "Compactness"},
    {"name": "MuFidelity", "nickname": "Correctness"},
    {"name": "Sensitivity", "nickname": "Continuity"},
  ]

  const explainerNickname = [
    { "name": "GuidedGradCam", "nickname": "Guided Grad-CAM" },
    { "name": "IntegratedGradients", "nickname": "Integrated Gradients" },
    { "name": "KernelShap", "nickname": "kernelSHAP" },
    { "name": "LRP", "nickname": "LRP" },
    { "name": "Lime", "nickname": "LIME" },
    { "name": "RAP", "nickname": "RAP" },
  ];
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const [experimentResults, setExperimentResults] = React.useState<ExperimentResult[]>([]);
  

  useEffect(() => {
    fetchExperiment(projectId, experiment).then((response) => {
      response = preprocess(response);
      const experimentResults = response.data.data
      experimentResults.forEach((experimentResult: ExperimentResult) => {
        experimentResult.explanations.sort((a, b) => a.rank - b.rank);
      });
      setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
      setLoading(false);
    }).catch((err) => {
      console.log(err);
    })
  }, [projectId])

  
  
  useEffect(() => {
    if (!(inputs.length > 0 && explainers.length > 0))
      return

    RunExperiment(projectId, experiment, {
      inputs: inputs,
      explainers: explainers,
      metrics: metrics
    }).then((response) => {
      response = preprocess(response);
      const experimentResults = response.data.data
      experimentResults.forEach((experimentResult: ExperimentResult) => {
        experimentResult.explanations.sort((a, b) => a.rank - b.rank);
      });
      setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
      setLoading(false);
    }).catch((err) => {
      console.log(err);
    })
  }, [inputs, explainers])

  if (loading) {
    return (
      <Box sx={{ mt: 15, display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <CircularProgress />
      </Box>
    )
  }
  

  return (
    <Box sx={{ mt: 15 }}>
      {experimentResults.map((result, index) => {
        return (
          <Box key={index}>
            {/* Image Cards */}
            <ImageList sx={{
              width: '100%', 
              minheight: '450px', 
              gap: 20, // Adjust the gap size here
              display: 'flex',
              flexDirection: 'row',
              overflowY: 'hidden', // Prevent vertical overflow
              overflowX: 'auto' // Allow horizontal scrolling
              }}>
              <ImageListItem key={0} sx={{ width: '200px', minHeight: "300px" }}>
                <Box sx={{ pt : 1}}>
                  <Plot
                    data={[result.input.data[0]]}
                    layout={result.input.layout}
                  />
                  <Typography variant="subtitle1" align="center"> Original </Typography>
                </Box>
                <Box sx={{ width: '100%' }}>
                  <Grid container spacing={2} alignItems="center" justifyContent="center">
                    <Grid item xs={12}>
                      <Typography variant="body2" align="center">True Label: {result.target}</Typography>
                    </Grid>
                    <Grid item xs={12}>
                      <Divider variant="middle" />
                      <Typography variant="body2" align="center">Top 3 Predictions</Typography>
                    </Grid>
                    {result.outputs.map((prob, index) => (
                      <Grid item xs={12} key={index}>
                        <Typography variant="body2" align="center">
                          {prob.key}: {(prob.value * 100).toFixed(2)}%
                        </Typography>
                      </Grid>
                    ))}
                    <Grid item xs={12}>
                      <Divider variant="middle" />
                      <Typography 
                        sx={{ color: result.target === result.outputs[0].key ? 'green' : 'red' }} 
                        variant="body2" 
                        align="center">
                        IsCorrect: {result.target === result.outputs[0].key ? 'True' : 'False'}
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              </ImageListItem>

              {result.explanations.map((exp, index) => {
                return exp.data && 
                  <ImageListItem key={index+1} sx={{ width: '200px', minHeight: "300px" }}>
                    <Box sx={{ p: 1 }}>
                      <Plot 
                        data={[exp.data.data[0]]}
                        layout={exp.data.layout}
                        />
                      <Typography variant="subtitle1" align="center">
                        {explainerNickname.find(n => n.name === exp.explainer)?.nickname}
                        </Typography>
                      {(Object.keys(exp.evaluation).length > 0) && (
                        <Typography variant="body2" sx={{ textAlign: 'center' }}> Rank {index+1}</Typography>
                      )}
                      
                      {Object.entries(exp.evaluation).map(([key, value]) => {
                        return (
                          <Typography key={key} variant="body2" sx={{ textAlign: 'center' }}> {nickname.find(n => n.name === key)?.nickname} ({value.toFixed(3)}) </Typography>
                      )})}
                    </Box>
                  </ImageListItem>
                }
              )}
            </ImageList>
          </Box>
        );
      })}
    </Box>
  );
}

export default Visualizations;

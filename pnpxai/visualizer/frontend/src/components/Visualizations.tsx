// src/components/Visualizations.tsx
import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { Card, CardContent, Typography, Box, 
  LinearProgress, ImageList, ImageListItem,
  Dialog, DialogContent, CircularProgress
} from '@mui/material';
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
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const [experimentResults, setExperimentResults] = React.useState<ExperimentResult[]>([]);
  

  useEffect(() => {
    const fetchExperimentResults = async () => {
      try {
        let response = await fetchExperiment(projectId, experiment);
        response = preprocess(response);
        const experimentResults = response.data.data
        experimentResults.forEach((experimentResult: ExperimentResult) => {
          experimentResult.explanations.sort((a, b) => a.rank - b.rank);
        });
        setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
        setLoading(false);
      }
      catch (err) {
        console.log(err);
      }
    }

    fetchExperimentResults();
  }
  , [projectId])

  
  
  useEffect(() => {
    const runExperimentResults = async () => {
      try {
        let response = await RunExperiment(projectId, experiment,
          {
            inputs: inputs,
            explainers: explainers,
            metrics: metrics
          }
          );
          response = preprocess(response);
          const experimentResults = response.data.data
          experimentResults.forEach((experimentResult: ExperimentResult) => {
            experimentResult.explanations.sort((a, b) => a.rank - b.rank);
          });
          setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
          setLoading(false);
        }
      catch (err) {
        console.log(err);
      }
    }
    
    if (inputs.length > 0 && explainers.length > 0) {
      runExperimentResults();
    }
  }
  , [inputs, explainers])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <CircularProgress />
      </Box>
    )
  }
  

  return (
    <Box sx={{ mt: 4 }}>
      {experimentResults.map((result, index) => {
        return (
          <Box key={index} sx={{ marginBottom: 4, paddingBottom: 2, borderBottom: '2px solid #e0e0e0' }}>
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
                <Box sx={{}}>
                  <Box sx={{}}>
                    <Typography variant="body2" align="center"> True Label : {result.target} </Typography>
                  </Box>
                  <Box sx={{}}>
                    <Typography variant="body2" align="center"> Predictions </Typography>
                  {result.outputs.map((prob, index) => (
                    <Typography variant="body2" align="center" key={index}> {prob.key} : {(prob.value*100).toFixed(2)}%</Typography>
                  ))}
                  </Box>
                  <Box sx={{}}>
                  <Typography 
                    sx={{color : result.target === result.outputs[0].key ? 'green' : 'red'}} 
                    variant="body2" 
                    align="center"> IsCorrect : {result.target === result.outputs[0].key ? 'True' : 'False'} </Typography>
                  </Box>
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
                      <Typography variant="subtitle1" align="center">{exp.explainer}</Typography>
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

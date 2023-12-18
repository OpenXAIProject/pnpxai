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
  const projectId = useSelector((state: RootState) => {
    return state.projects.data[0].id;
  });
  const [experimentResults, setExperimentResults] = React.useState<ExperimentResult[]>([]);
  

  useEffect(() => {
    const fetchExperimentResults = async () => {
      try {
        let response = await fetchExperiment(projectId, experiment);
        response = preprocess(response);
        AddMockData(response); // Add mock data for testing
        const experimentResults = response.data.data
        setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
        setLoading(false);
      }
      catch (err) {
        console.log(err);
      }
    }

    fetchExperimentResults();
  }
  , [])

  
  
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
          AddMockData(response); // Add mock data for testing
          const experimentResults = response.data.data
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
            {/* Info Cards */}
            {/* <Box sx={{ display: 'flex', justifyContent: 'space-around', marginBottom: 2 }}> */}
              {/* Label Card */}
              {/* <Card sx={{ minWidth: 275 }}>
                <CardContent>
                  <Typography variant="h5" component="div">True Label</Typography>
                  <Typography variant="body2">{result.prediction.label}</Typography>
                </CardContent>
              </Card> */}

              {/* Probability Card */}
              {/* <Card sx={{ minWidth: 275 }}>
                <CardContent>
                  <Typography variant="h5" component="div">Probabilities</Typography>
                  {result.prediction.probPredictions.map((prob, index) => (
                    <Box key={index} sx={{ mb: 1 }}>
                      <Typography variant="body2">{prob.label}: {prob.score}%</Typography>
                      <LinearProgress variant="determinate" value={prob.score} />
                    </Box>
                  ))}
                </CardContent>
              </Card> */}

              {/* Result Card */}
              {/* <Card sx={{ minWidth: 275, bgcolor: result.prediction.isCorrect ? 'lightgreen' : 'red' }}>
                <CardContent>
                  <Typography variant="h5" component="div">{result.prediction.isCorrect ? 'Correct' : 'False'}</Typography>
                </CardContent>
              </Card> */}
            {/* </Box> */}

            {/* Image Cards */}
            <ImageList sx={{
              width: '100%', 
              height: '400px', 
              gap: 20, // Adjust the gap size here
              rowHeight: 164,
              display: 'flex',
              flexDirection: 'row',
              overflowY: 'hidden', // Prevent vertical overflow
              overflowX: 'auto' // Allow horizontal scrolling
              }}>
              <ImageListItem key={0} sx={{ width: '240px', minHeight: "300px" }}>
                <Box sx={{ p: 1}}>
                  <Plot
                    data={[result.input.data[0]]}
                    layout={result.input.layout}
                  />
                  <Typography variant="subtitle1" align="center"> Original </Typography>
                </Box>
                <Box sx={{ p: 1}}>
                  <Box sx={{ p : 1 }}>
                    <Typography variant="body2" align="center"> True Label : {result.prediction.label} </Typography>
                  </Box>
                  <Box sx={{ p : 1 }}>
                  {result.prediction.probPredictions.map((prob, index) => (
                    <Typography variant="body2" align="center" key={index}>{prob.label}: {prob.score}%</Typography>
                  ))}
                  </Box>
                  <Box sx={{ p : 1 }}>
                  <Typography 
                    sx={{color : result.prediction.isCorrect ? 'green' : 'red'}} 
                    variant="body2" 
                    align="center"> IsCorrect : {result.prediction.isCorrect ? 'True' : 'False'} </Typography>
                  </Box>
                </Box>
              </ImageListItem>

              {result.explanations.map((exp, index) => {
                return exp.data && 
                  <ImageListItem key={index+1} sx={{ width: '240px', minHeight: "300px" }}>
                    <Box sx={{ p: 1 }}>
                      <Plot 
                        data={[exp.data.data[0]]}
                        layout={exp.data.layout}
                        />
                      <Typography variant="subtitle1" align="center">{exp.explainer}</Typography>
                      <Typography variant="body2" sx={{ textAlign: 'center' }}> Rank {index+1}</Typography>
                      
                      {Object.entries(exp.evaluation).map(([key, value]) => {
                        return (
                          <Typography key={key} variant="body2" sx={{ textAlign: 'center' }}> {key} ({value.toFixed(3)}) </Typography>
                      )})}
                      
                      <Typography variant="body2" sx={{ textAlign: 'center' }}> Weighted score ({exp.weighted_score.toFixed(3)})</Typography>
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

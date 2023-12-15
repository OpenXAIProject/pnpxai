// src/components/Visualizations.tsx
import React, { useEffect } from 'react';
import { Card, CardContent, Typography, Box, 
  LinearProgress, ImageList, ImageListItem,
  Dialog, DialogContent, CircularProgress
} from '@mui/material';
import Plot from 'react-plotly.js';
import { ExperimentResult } from '../app/types';
import { fetchExperiment } from '../features/apiService';
import { preprocess, AddMockData } from './utils';


// 수정 방안
// ExperimentComponent에서 Run Experiment 버튼을 누르면
// Visualization 내부에서 experimentResult를 Fetch하도록 수정
// ExperimentComponent에서는 fetch를 하기 위한 input data를 만들어서 보내주기만 하면 됨


const Visualizations: React.FC<{ inputs: number[]; explainers: number[]; setLoading: any}> = ({ inputs, explainers, setLoading }) => {
  const [experimentResults, setExperimentResults] = React.useState<ExperimentResult[]>([]);
  

  useEffect(() => {
    const fetchExperimentResults = async () => {
      try {
        const response = await fetchExperiment(
          'test_project',
          'test_experiment',
          {
            inputs: inputs,
            explainers: explainers
          }
          );
          preprocess(response);
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
      fetchExperimentResults();
    }
  }
  , [inputs, explainers]);
  

  return (
    <Box sx={{ mt: 4 }}>
      {experimentResults.map((result, index) => {
        return (
          <Box key={index} sx={{ marginBottom: 4, paddingBottom: 2, borderBottom: '2px solid #e0e0e0' }}>
            {/* Info Cards */}
            <Box sx={{ display: 'flex', justifyContent: 'space-around', marginBottom: 2 }}>
              {/* Label Card */}
              <Card sx={{ minWidth: 275 }}>
                <CardContent>
                  <Typography variant="h5" component="div">True Label</Typography>
                  <Typography variant="body2">{result.prediction.label}</Typography>
                </CardContent>
              </Card>

              {/* Probability Card */}
              <Card sx={{ minWidth: 275 }}>
                <CardContent>
                  <Typography variant="h5" component="div">Probabilities</Typography>
                  {result.prediction.probPredictions.map((prob, index) => (
                    <Box key={index} sx={{ mb: 1 }}>
                      <Typography variant="body2">{prob.label}: {prob.score}%</Typography>
                      <LinearProgress variant="determinate" value={prob.score} />
                    </Box>
                  ))}
                </CardContent>
              </Card>

              {/* Result Card */}
              <Card sx={{ minWidth: 275, bgcolor: result.prediction.isCorrect ? 'lightgreen' : 'red' }}>
                <CardContent>
                  <Typography variant="h5" component="div">{result.prediction.isCorrect ? 'Correct' : 'False'}</Typography>
                </CardContent>
              </Card>
            </Box>

            {/* Image Cards */}
            <ImageList sx={{
              width: '100%', 
              height: '300px', 
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
              </ImageListItem>

              {result.visualizations.map((viz, index) => {
                return (
                  <ImageListItem key={index+1} sx={{ width: '240px', minHeight: "300px" }}>
                      <Box sx={{ p: 1 }}>
                        <Plot 
                          data={[viz.data.data[0]]}
                          layout={viz.data.layout}
                          />
                        <Typography variant="subtitle1" align="center">{viz.explainer}</Typography>
                        <Typography variant="body2" sx={{ textAlign: 'center' }}> Rank {index+1}</Typography>
                        <Typography variant="body2" sx={{ textAlign: 'center' }}> Faithfulness ({viz.metrics.faithfulness})</Typography>
                        <Typography variant="body2" sx={{ textAlign: 'center' }}> Robustness ({viz.metrics.robustness})</Typography>
                      </Box>
                  </ImageListItem>
                );
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

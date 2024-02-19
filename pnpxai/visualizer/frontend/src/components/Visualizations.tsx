// src/components/Visualizations.tsx
import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { 
  Typography, Box, ImageList, ImageListItem, 
  CircularProgress, Grid, Divider,Stack
} from '@mui/material';
import LinearProgress from '@mui/material/LinearProgress';
import Plot from 'react-plotly.js';
import { ExperimentResult } from '../app/types';
import { RunExperiment, fetchExperiment, fetchExperimentStatus } from '../features/apiService';
import { preprocess } from './utils';
import { ErrorProps, ErrorSnackbar } from './ErrorSnackBar';
import { nickname, explainerNickname } from './util';


const Visualizations: React.FC<{ 
  experiment: string; inputs: number[]; explainers: number[]; metrics: number[]; loading: boolean; setLoading: any
}> = ({ experiment, inputs, explainers, metrics, loading, setLoading }) => {
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const colorScale = useSelector((state: RootState) => state.projects.config.colorMap);
  const [experimentResults, setExperimentResults] = React.useState<ExperimentResult[]>([]);
  const [isError, setIsError] = React.useState<boolean>(false);
  const [errorInfo, setErrorInfo] = React.useState<ErrorProps[]>([]);
  const [progress, setProgress] = React.useState(0);
  const [progressMsg, setProgressMsg] = React.useState("Loading...");
  

  useEffect(() => {
    fetchExperiment(projectId, experiment).then((response) => {
      if (response.data.errors.length === 0) {
        response = preprocess(response, {colorScale: colorScale});
        const experimentResults = response.data.data
        experimentResults.forEach((experimentResult: ExperimentResult) => {
          experimentResult.explanations.sort((a, b) => a.rank - b.rank);
        });
        setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
      } else {
        setIsError(true);
        setErrorInfo(response.data.errors);
      }
      
    }).catch((err) => {
      console.log(err);
      setIsError(true);
      setErrorInfo(err.response.data.errors);
    }).finally(() => {
      setLoading(false);
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
      if (response.data.errors.length === 0) {
        response = preprocess(response, {colorScale: colorScale});
        const experimentResults = response.data.data
        experimentResults.forEach((experimentResult: ExperimentResult) => {
          experimentResult.explanations.sort((a, b) => a.rank - b.rank);
        });
        setExperimentResults(JSON.parse(JSON.stringify(experimentResults)));
      } else {
        setIsError(true);
        setErrorInfo(response.data.errors);
      }
    }).catch((err) => {
      console.log(err);
      setIsError(true);
      setErrorInfo(err.response.data.errors);
    }).finally(() => {
      setLoading(false);
      setProgress(0);
      setProgressMsg("Loading...");
    })
  }, [inputs, explainers, colorScale])

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
  
    if (loading) {
      interval = setInterval(async () => {
        fetchExperimentStatus(projectId, experiment)
        .then((response) => {
          setProgress(response.data.data.progress);
          setProgressMsg(response.data.data.message);
        }).catch((err) => {
          console.log(err);
        });
      }, 1000);
    } else {
    // If loading is false, clear the interval if it exists
    if (interval) {
      clearInterval(interval);
    }
    }
  
    // Cleanup function to clear the interval when the component unmounts or before re-running the effect
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [loading]); // Dependency array includes `loading`, so the effect re-runs when `loading` changes

  if (loading) {
    return (
      <Box sx={{ mt : 5 }}>
        <Stack direction="column" spacing={2} alignItems="center">
          <CircularProgress />
          <Typography variant="h4">{progressMsg}</Typography>
          <Stack sx={{ minWidth : `400px`}} direction="row" alignItems="center">
            <LinearProgress sx={{ width: `100%`}} variant="determinate" value={progress*100} />
            <Typography variant="h6" sx={{ ml: 2 }}>{(progress*100).toFixed(0)}%</Typography>
          </Stack>
        </Stack>
      </Box>
    )
  }

  if (isError) {
    return (
      <Box sx={{ mt: 15 }}>
        {errorInfo.map((error, index) => (
          <ErrorSnackbar key={index} name={error.name} message={error.message} trace={error.trace} />
        ))}
      </Box>
    );
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
                          <Typography key={key} variant="body2" sx={{ textAlign: 'center' }}> {nickname.find(n => n.name === key)?.nickname} ({value?.toFixed(3) ?? ''}) </Typography>
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

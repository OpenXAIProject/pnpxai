// src/components/Visualizations.tsx
import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { Typography, Box, ImageList, ImageListItem, CircularProgress, Grid, Divider,
  Alert, AlertTitle, Snackbar
} from '@mui/material';
import Plot from 'react-plotly.js';
import { ExperimentResult } from '../app/types';
import { RunExperiment, fetchExperiment } from '../features/apiService';
import { preprocess, AddMockData } from './utils';

interface ErrorProps {
  name: string;
  message: string;
  trace?: string;
}

const ErrorSnackbar: React.FC<ErrorProps> = ({ name, message, trace }) => {
  const [open, setOpen] = React.useState(true);

  const handleClose = () => setOpen(false);

  const addTraceTitle = () => {
    if (trace) {
      return (
        <AlertTitle>
          {name}: {message}
        </AlertTitle>
      );
    } else {
      return <AlertTitle>{name}</AlertTitle>;
    }
  };

  const renderTrace = (trace: string) => {
    console.log(trace);
    console.log(trace.split("\\n"));
    return trace.slice(0, -2).split('\\n').map((line, index) => {
      let toPrint = line;
      if (index % 3 === 0) {
        toPrint = line.replace("'", "").replace(",", "").replace(" '", "").replace("[", "");
      }

      return (
        <pre key={index}>
          {toPrint}
        </pre>
      );
    });
  };


  return (
    <Snackbar anchorOrigin={{ vertical : 'top', horizontal : 'right' }} open={open} onClose={handleClose}>
      <Alert severity="error" onClose={handleClose}>
        {addTraceTitle()}
        {trace && (
          renderTrace(trace)
        )}
      </Alert>
    </Snackbar>
  );
};

const Visualizations: React.FC<{ 
  experiment: string; inputs: number[]; explainers: number[]; metrics: number[]; loading: boolean; setLoading: any
}> = ({ experiment, inputs, explainers, metrics, loading, setLoading }) => {
  // TODO: change this nickname to the real name
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
  const [isError, setIsError] = React.useState<boolean>(false);
  const [errorInfo, setErrorInfo] = React.useState<ErrorProps[]>([]);
  

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
      if (response.data.errors.length === 0) {
        response = preprocess(response);
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
    }).finally(() => {
      setLoading(false);
    })
  }, [inputs, explainers])

  if (loading) {
    return (
      <Box sx={{ mt: 15, display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <CircularProgress />
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

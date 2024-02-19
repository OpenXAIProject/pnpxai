// src/components/ExperimentComponent.tsx
import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { 
  Grid, Box, Typography, 
  Button, Chip, 
  Dialog, DialogActions, DialogContent, DialogTitle, 
  Alert, FormControlLabel,
  Checkbox, Paper, Card, Tooltip, CardContent, CircularProgress
} from '@mui/material';
import Visualizations from './Visualizations';
import { Experiment, Metric } from '../app/types';
import { fetchInputsByExperimentId } from '../features/apiService';
import { ErrorProps, ErrorSnackbar } from './modal/ErrorSnackBar';
import GalleryModal from './modal/GalleryModal';
import { InputData } from '../app/types';
import { nickname, domain_extension_plan } from './util';

const ExperimentComponent: React.FC<{experiment: Experiment, key: string}> = ( {experiment} ) => {
  // Basic Data
  const projectId = useSelector((state: RootState) => state.global.status.currentProject);
  const defaultMetrics = experiment.metrics.filter(metric => metric.name === 'MuFidelity');
  const sortedExplainers = [...experiment.explainers].sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase())
  );
  

  // User Input
  const [inputs, setInputs] = useState<string[]>([]);
  const [selectedInputs, setSelectedInputs] = useState<number[]>([]);
  const [explainers, setExplainers] = useState<number[]>(experiment.explainers.map(explainer => explainer.id));
  const [selectedExplainers, setSelectedExplainers] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>(defaultMetrics ? defaultMetrics : []) ;
  const [selectedMetrics, setSelectedMetrics] = useState<number[]>([]);

  const [isModalOpen, setIsModalOpen] = useState(false);

  const [isError, setIsError] = useState(false);
  const [errorInfo, setErrorInfo] = useState<ErrorProps[]>([]);
  const [loading, setLoading] = React.useState<boolean>(false);
  const [showDialog, setShowDialog] = React.useState(false);
  const [galleryInputs, setGalleryInputs] = useState<InputData[]>([]);

  useEffect(() => {
    if (isModalOpen) {
      fetchInputsByExperimentId(projectId, experiment.id)
      .then(response => {
        setGalleryInputs(
          response.data.data.map((input: string, index: number) => {
            const parsedInput = JSON.parse(input);
            return {
              id: index.toString(),
              source: parsedInput.data[0].source,
            };
          })
        )
      })
      .catch(error => {
        console.error(error);
        setIsError(true);
        setErrorInfo(error.response.data.errors);
      });
    }
  }, [isModalOpen]);

  
  const handleChipCancel = (imageId: string) => {
    setInputs(inputs.filter(input => input !== imageId));
  }

  const handleAlgorithmCheckboxChange = (explainerId: number, isChecked: boolean) => {
    if (isChecked) {
      setExplainers([...explainers, explainerId]);
    } else {
      setExplainers(explainers.filter(id => id !== explainerId));
    }
  };

  const handleCheckboxChange = (item: Metric, isChecked: boolean) => {
    if (isChecked) {
      setMetrics(prevItems => [...prevItems, item]);
    } else {
      setMetrics(prevItems => prevItems.filter(i => i.id !== item.id));
    }
  };


  const handleRunExperiment = () => {
    if (inputs.length === 0 || explainers.length === 0) {
      setShowDialog(true);
      return;
    }

    setLoading(true);
    setSelectedInputs(inputs.map(input => Number(input)));
    setSelectedExplainers(explainers.map(explainer => Number(explainer)));
    setSelectedMetrics(metrics.map(metric => metric.id));
  };

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
    <Box sx={{ m: 1 }}>
      <Box sx={{ mt: 3, mb: 3, ml: 1, pb: 3}}>
      <Card>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Box sx={{ pl:4, p: 2, borderBottom: 1, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
              <Tooltip 
                title={(
                  <Card>
                    <CardContent>
                      <Typography variant="body1"> {domain_extension_plan} </Typography>
                    </CardContent>
                  </Card>
                )}
                >
                  <Box>
                    <Typography variant="h6"> Task </Typography>
                    <Typography variant="body1"> Image Classification </Typography>
                  </Box>
              </Tooltip>
              <Box>
                <Typography variant="h6"> Experiment Name </Typography>
                <Typography variant="body1"> {experiment.name} </Typography>
              </Box>
              <Box>
                <Typography variant="h6"> Model Name </Typography>
                <Typography variant="body1"> {experiment.model.name} </Typography>
              </Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={2} sx={{borderRight: 1, borderColor: 'divider'}}>
            {/* Sidebar */}
            <Box sx={{ m: 1 }}>
              {/* Images Box */}
              <Box sx={{ ml: 1, mr : 1, borderBottom: 1, borderColor: 'divider', p: 1 }}>
                <Typography variant="h6"> Select Instance </Typography>
                <Button variant="contained" color="primary" onClick={() => setIsModalOpen(true)} sx={{ mt: 2 }}> Show Instances</Button>
                <Box sx={{ mt: 2 }}>
                  {inputs.map((image, index) => (
                    <Chip key={index} label={image} onDelete={() => handleChipCancel(image)} sx={{ mt: 1 }} />
                  ))}
                </Box>
              </Box>

              {/* Explainer Box */}
              <Box sx={{ ml: 1, mr : 1, borderBottom: 1, borderColor: 'divider', p: 1}}>
                <Typography variant="h6">Select Explainers</Typography>
                {sortedExplainers.map((explainerObj, index) => (
                  <FormControlLabel
                    key={explainerObj.id}
                    control={
                      <Checkbox
                        checked={explainers.includes(explainerObj.id)}
                        onChange={(e) => handleAlgorithmCheckboxChange(explainerObj.id, e.target.checked)}
                      />
                    }
                    label={explainerObj.name}
                    sx={{ fontSize: '0.75rem' }} // Smaller font size for labels
                  />
                ))}
              </Box>
                
              {/* Metrics Box */}
              <Box sx={{ ml: 1, mr : 1, borderBottom: 1, borderColor: 'divider', p: 1 }}>
                <Typography variant="h6"> Select Evaluation Metrics </Typography>
                {experiment.metrics.map(item => (
                  <FormControlLabel
                    key={item.id}
                    control={
                      <Checkbox
                        checked={metrics.some(i => i.id === item.id)}
                        onChange={(e) => handleCheckboxChange(item, e.target.checked)}
                      />
                    }
                    label={nickname.find(n => n.name === item.name)?.nickname}
                  />
                ))}
              </Box>
              
              {/* Run Experiment Button */}
              <Box sx={{ ml: 1, mr : 1, p: 1 }}>
                {/* <Button variant="contained" color="primary" onClick={handleAutoExplain} sx={{ mt: 2 }}>Auto Explain</Button> */}
                <Button variant="contained" color="primary" onClick={handleRunExperiment} sx={{ mt: 2 }}>Run Experiment</Button>
              </Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={10}>
            {/* Experiment Visualization */}
            <Box sx={{ mt: 2, pl: 2 }}>
              <Visualizations
                experiment={experiment.name}
                inputs={selectedInputs}
                explainers={selectedExplainers}
                metrics={selectedMetrics}
                loading={loading}
                setLoading={setLoading}
              />
            </Box>
          </Grid>
        </Grid>
      </Card>
      
      {/* Image Selection Dialog */}
      <GalleryModal
        isModalOpen={isModalOpen}
        setIsModalOpen={setIsModalOpen}
        galleryInputs={galleryInputs}
        setInputs={setInputs}
      />
      
      
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Dialog 
          open={showDialog}
          onClose={() => setShowDialog(false)}
          >
          <DialogContent>
            <Alert severity='info'>Please select at least 1 input and 1 explainer and 1 evaluation metric </Alert>
          </DialogContent>
        </Dialog>
      </Box>
      </Box>                  
    </Box>
  );
}

export default ExperimentComponent;




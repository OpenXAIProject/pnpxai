// src/components/ExperimentComponent.tsx
import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../app/store';
import { 
  Grid, Box, Typography, 
  Button, Chip, 
  Dialog, DialogContent, 
  Alert, FormControlLabel,
  Checkbox, Card, Tooltip, CardContent
} from '@mui/material';
import Visualizations from './Visualizations';
import { Experiment, InputData, Explainer, Metric } from '../app/types';
import GalleryModal from './modal/GalleryModal';
import { nickname, domain_extension_plan } from './util';
import { 
  setInputs,
  setExplainers,
  setMetrics,
} from '../features/globalState';

const ExperimentComponent: React.FC<{experiment: Experiment, key: string}> = ( {experiment} ) => {
  // Set current experiment
  const dispatch = useDispatch();
  const expId = experiment.id;

  // Basic Data
  const projects = useSelector((state: RootState) => state.global.projects);
  const projectId = useSelector((state: RootState) => state.global.status.currentProject);
  const expCache = useSelector((state: RootState) => state.global.expCache.filter((item) => item.projectId === projectId && item.expId === expId)[0]);

  const galleryInputs = expCache?.galleryInputs
  const selectedInputs = expCache?.inputs;
  const selectedExplainers = expCache?.explainers;
  const selectedMetrics = expCache?.metrics;

  const explainerOptions = [...(projects.find(project => project.id === projectId)
    ?.experiments.find(exp => exp.id === expId)
    ?.explainers || [])] // Cloning the array
    .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));

  const metricOptions = projects.find(project => project.id === projectId)
    ?.experiments.find(exp => exp.id === expId)
    ?.metrics;

  const [visualizationConfig, setVisualizationConfig] = useState({
    inputs: [] as InputData[],
    explainers: [] as Explainer[],
    metrics: [] as Metric[],
  });
  
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [showDialog, setShowDialog] = React.useState(false);

  const handleChipCancel = (imageId: string) => {
    dispatch(setInputs({ projectId: projectId, expId: expId, inputs: selectedInputs.filter(input => input.id !== imageId) }));
  }

  const handleExplainerCheckBox = (item: Explainer, isChecked: boolean) => {
    if (isChecked) {
      dispatch(setExplainers({ projectId: projectId, expId: expId, explainers: [...selectedExplainers, item] }));
    } else {
      dispatch(setExplainers({ projectId: projectId, expId: expId, explainers: selectedExplainers.filter(explainer => explainer.id !== item.id) }));
    }
  };

  const handleMetricCheckBox = (item: Metric, isChecked: boolean) => {
    if (isChecked) {
      dispatch(setMetrics({ projectId: projectId, expId: expId, metrics: [...selectedMetrics, item] }));
    } else {
      dispatch(setMetrics({ projectId: projectId, expId: expId, metrics: selectedMetrics.filter(metric => metric.id !== item.id) }));
    }
  };


  const handleRunExperiment = () => {
    if (selectedInputs.length === 0 || selectedExplainers.length === 0) {
      setShowDialog(true);
      return;
    }

    setVisualizationConfig({
      inputs: selectedInputs,
      explainers: selectedExplainers,
      metrics: selectedMetrics,
    });
  };

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
                  {selectedInputs.map((input, index) => (
                    <Chip key={index} label={input.id} onDelete={() => handleChipCancel(input.id)} sx={{ mt: 1 }} />
                  ))}
                </Box>
              </Box>

              {/* Explainer Box */}
              <Box sx={{ ml: 1, mr : 1, borderBottom: 1, borderColor: 'divider', p: 1}}>
                <Typography variant="h6">Select Explainers</Typography>
                {explainerOptions && explainerOptions.map(explainer => (
                  <FormControlLabel
                    key={explainer.id}
                    control={
                      <Checkbox
                        checked={selectedExplainers.some(item => item.name === explainer.name)}
                        onChange={(e) => handleExplainerCheckBox(explainer, e.target.checked)}
                      />
                    }
                    label={explainer.name}
                    sx={{ fontSize: '0.75rem' }} // Smaller font size for labels
                  />
                ))}
              </Box>
                
              {/* Metrics Box */}
              <Box sx={{ ml: 1, mr : 1, borderBottom: 1, borderColor: 'divider', p: 1 }}>
                <Typography variant="h6"> Select Evaluation Metrics </Typography>
                {metricOptions && metricOptions.map(metric => (
                  <FormControlLabel
                    key={metric.id}
                    control={
                      <Checkbox
                        checked={selectedMetrics.some(item => item.name === metric.name)}
                        onChange={(e) => handleMetricCheckBox(metric, e.target.checked)}
                      />
                    }
                    label={nickname.find(n => n.name === metric.name)?.nickname}
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
                experiment={experiment}
                inputs={visualizationConfig.inputs}
                explainers={visualizationConfig.explainers}
                metrics={visualizationConfig.metrics}
              />
            </Box>
          </Grid>
        </Grid>
      </Card>
      
      {/* Image Selection Dialog */}
      <GalleryModal
        experiment={experiment}
        isModalOpen={isModalOpen}
        setIsModalOpen={setIsModalOpen}
        galleryInputs={galleryInputs}
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




// src/components/ExperimentComponent.tsx
import React, { useState, useEffect } from 'react';
import { 
  Grid, Box, Typography, 
  Button, Chip, 
  Dialog, DialogActions, DialogContent, DialogTitle, 
  Alert, FormControlLabel,
  Checkbox, Paper, Card, Tooltip, CardContent, tooltipClasses
} from '@mui/material';
import Visualizations from './Visualizations';
import { Experiment, Metric } from '../app/types';

const ExperimentComponent: React.FC<{experiment: Experiment, key: number}> = ( {experiment} ) => {
  // TODO: change this nickname to the real name
  const nickname = [
    {"name": "Complexity", "nickname": "Compactness"},
    {"name": "MuFidelity", "nickname": "Correctness"},
    {"name": "Sensitivity", "nickname": "Continuity"},
  ]
  const domain_extension_plan = "The task will be extended to other domains(Tabular, Text, Time Series) in the future."
  
  // Basic Data
  const galleryInputs = experiment.inputs.map(input => {
    return {id: input.id, source: input.imageObj.data[0].source}
  });


  // User Input
  const [inputs, setInputs] = useState<number[]>([]);
  const [selectedInputs, setSelectedInputs] = useState<number[]>([]);
  const [explainers, setExplainers] = useState<number[]>(experiment.explainers.map(explainer => explainer.id));
  const [selectedExplainers, setSelectedExplainers] = useState<number[]>([]);
  const defaultMetrics = experiment.metrics.filter(metric => metric.name === 'MuFidelity');
  const [metrics, setMetrics] = useState<Metric[]>(defaultMetrics ? defaultMetrics : []) ;
  const [selectedMetrics, setSelectedMetrics] = useState<number[]>([]);
  const sortedExplainers = [...experiment.explainers].sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase())
  );



  // const inputs: InputData[] = JSON.parse(JSON.stringify(experiment.inputs));
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [tmpInputs, setTmpInputs] = useState<number[]>([]); // State to track selection in the modal

  const [isExperimentRun, setIsExperimentRun] = useState(false); // New state to track if experiment is run
  const [loading, setLoading] = React.useState<boolean>(false);
  const [showDialog, setShowDialog] = React.useState(false);

  
  useEffect(() => {
    // Synchronize modalSelection with selectedImages
    if (isModalOpen) {
      setTmpInputs(inputs);
    }
  }, [isModalOpen, inputs]);


  
  // Image Selection Handlers
  // Modal Handlers
  const handleImageClick = (imageId: number) => {
    // Only 1 Input.
    setTmpInputs([imageId]);

    // Multiple Inputs.
    // if (tmpInputs.includes(imageId)) {
    //   setTmpInputs(prevSelection => prevSelection.filter(item => item !== imageId));
    // } else {
    //   setTmpInputs(prevSelection => [...prevSelection, imageId]);
    // }
  };

  const handleConfirmSelection = () => {
    setInputs(tmpInputs);
    setIsModalOpen(false);
  };

  const handleCancelSelection = () => {
    setIsModalOpen(false);
  };

  const handleChipCancel = (imageId: number) => {
    setInputs(inputs.filter(input => input !== imageId));
  }


  // Algorithm Selection Handlers
  const addAlgorithm = (explainer: number) => {
    if (!explainers.includes(explainer)) {
      setExplainers([...explainers, explainer]);
    }
  };

  const removeAlgorithm = (explainer: number) => {
    setExplainers(explainers.filter(alg => alg !== explainer));
  };

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
    setIsExperimentRun(true);
    setSelectedInputs(inputs.map(input => Number(input)));
    setSelectedExplainers(explainers.map(explainer => Number(explainer)));
    setSelectedMetrics(metrics.map(metric => metric.id));
  };

  const handleAutoExplain = () => {
    // Select all explainer IDs
    const allExplainerIds = experiment.explainers.map(explainer => explainer.id);
    setExplainers(allExplainerIds);
  
    // Select all metrics
    const allMetrics = experiment.metrics;
    setMetrics(allMetrics);
  
    // Set selected states which are used in the Experiment run
    setSelectedInputs(inputs.map(input => Number(input)));
    setSelectedExplainers(allExplainerIds);
    setSelectedMetrics(allMetrics.map(metric => metric.id));
  
    setLoading(true);
    setIsExperimentRun(true);
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
      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Select Intances</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {galleryInputs.map((input, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Paper 
                  sx={{ 
                    height: "300px", 
                    display: 'flex', 
                    flexDirection: 'column', // Set the flex direction to column
                    justifyContent: 'center', // Aligns children vertically in the center
                    alignItems: 'center', // Aligns children horizontally in the center
                    cursor: 'pointer', 
                    opacity: tmpInputs.includes(input.id) ? 0.5 : 1
                  }}
                  onClick={() => handleImageClick(input.id)}
                >
                    <img src={input.source} width={240} height={200} alt={String(input.id)} />
                    <Typography variant="subtitle1" align="center">{input.id}</Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelSelection}>Cancel</Button>
          <Button onClick={handleConfirmSelection}>OK</Button>
        </DialogActions>
      </Dialog>
      
      
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




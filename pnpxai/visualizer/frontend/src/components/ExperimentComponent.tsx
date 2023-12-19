// src/components/ExperimentComponent.tsx
import React, { useState, useEffect } from 'react';
import { 
  Grid, Box, Typography, 
  Button, Chip, 
  Dialog, DialogActions, DialogContent, DialogTitle, 
  Alert, FormControlLabel,
  Checkbox, Paper
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import Visualizations from './Visualizations';
import { Experiment, Metric } from '../app/types';

const ExperimentComponent: React.FC<{experiment: Experiment, key: number}> = ( {experiment} ) => {
  // Basic Data
  const galleryInputs = experiment.inputs.map(input => {
    return {id: input.id, source: input.imageObj.data[0].source}
  });


  // User Input
  const [inputs, setInputs] = useState<number[]>([]);
  const [selectedInputs, setSelectedInputs] = useState<number[]>([]);
  const [explainers, setExplainers] = useState<number[]>([]);
  const [selectedExplainers, setSelectedExplainers] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>(experiment.metrics.filter(metric => metric.name === "Complexity"));
  const [selectedMetrics, setSelectedMetrics] = useState<number[]>(experiment.metrics.map(metric => metric.id));



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



  return (
    <Box sx={{ mt: 3, mb: 3, ml: 1, pb: 3, borderBottom: 1, minHeight: "600px" }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={2}>
          {/* Sidebar */}
          <Box sx={{ borderRight: 1, borderColor: 'divider', m: 2 }}>
            
            {/* Images Box */}
            <Box sx={{ mb: 3, borderBottom: 1, borderColor: 'divider', padding: 2 }}>
              <Typography variant="h6">Images</Typography>
              <Button variant="contained" color="primary" onClick={() => setIsModalOpen(true)} sx={{ mt: 2 }}>Select Images</Button>
              <Box sx={{ mt: 3 }}>
                {inputs.map((image, index) => (
                  <Chip key={index} label={image} onDelete={() => handleChipCancel(image)} sx={{ mt: 1 }} />
                ))}
              </Box>
            </Box>

            {/* Algorithms Box */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Algorithms</Typography>
              <Box sx={{ mt: 3, border: 1, borderColor: 'divider', padding: 1 }}>
                {experiment.explainers
                  .filter(explainerObj => explainers.includes(explainerObj.id))
                  .map((explainerObj, index) => (
                  <Chip 
                    key={index} 
                    label={explainerObj.name}
                    onDelete={() => removeAlgorithm(explainerObj.id)} 
                    deleteIcon={<CloseIcon />}
                    sx={{ m: 0.5 }}
                  />
                ))}
              </Box>
              <Box sx={{ mt: 2 }}>
                {experiment.explainers
                  .filter(explainerObj => !explainers.includes(explainerObj.id))
                  .map((explainerObj, index) => (
                    <Chip 
                      key={index} 
                      label={explainerObj.name} 
                      onClick={() => addAlgorithm(explainerObj.id)}
                      sx={{ m: 0.5 }}
                    />
                  ))}
              </Box>
            </Box>
              
              {/* Metrics Box */}
              <Box sx={{ mt : 3}}>
                <Typography variant="h6"> Metrics </Typography>
                {experiment.metrics.map(item => (
                  <FormControlLabel
                    key={item.id}
                    control={
                      <Checkbox
                        checked={metrics.some(i => i.id === item.id)}
                        onChange={(e) => handleCheckboxChange(item, e.target.checked)}
                      />
                    }
                    label={item.name}
                  />
                ))}
              </Box>
          </Box>
        </Grid>
        <Grid item xs={12} md={10}>
          {/* Experiment Visualization */}
          <Box sx={{ pl: 2 }}>
            <Typography variant="h5">{experiment?.name}</Typography>
            <Button variant="contained" color="secondary" onClick={handleRunExperiment} sx={{ mt: 2 }}>Run Experiment</Button>
            {/* Experiment Visualization */}
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
      
      
      {/* Image Selection Dialog */}
      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Select Images</DialogTitle>
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
            <Alert severity='info'>Please select at least 1 input and 1 explainer </Alert>
          </DialogContent>
        </Dialog>
      </Box>
      
    </Box>
  );
}

export default ExperimentComponent;




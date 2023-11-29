import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import { 
  Container, Grid, Box, Typography, 
  Button, Chip, 
  Dialog, DialogActions, DialogContent, DialogTitle, 
  List, ListItem, ListItemText, CircularProgress,
  Checkbox, Paper
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';
import Visualizations from './Visualizations';


interface Props {
  id: number;
}

const ExperimentComponent: React.FC<Props> = ({ id }) => {
  function makePredictions(images: string[]) {
    const predictions = [];
    for (const image of images) {
      predictions.push({
        image,
        label : "cat",
        probPredictions: [
          { label: 'cat', score: 0.9 },
          { label: 'dog', score: 0.05 },
          { label: 'bird', score: 0.05 },
        ],
        isCorrect: true,
      });
    }
    return predictions;
  }

  function makeEvaluations(images: string[], algorithms: string[]) {
    const evaluations = [];
    for (const image of images) {
      for (const algorithm of algorithms) {
        evaluations.push({
          image,
          algorithm,
          evaluation: {
            faithfulness: 0.6,
            robustness: 0.4,
          },
        });
      }
    }
    return evaluations;
  }

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [modalSelection, setModalSelection] = useState<string[]>([]); // State to track selection in the modal
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [experimentOutput, setExperimentOutput] = useState<string>('');
  const [predictions, setPredictions] = useState<any[]>([]);
  const [evaluations, setEvaluations] = useState<any[]>([]);
  const [isExperimentRun, setIsExperimentRun] = useState(false); // New state to track if experiment is run
  const [loading, setLoading] = useState(false); // State for loading modal

  const experimentData = useSelector((state: RootState) => {
    return state.experiments.data.find((experiment) => experiment.experiment_id === id);
  });
  
  useEffect(() => {
    // Synchronize modalSelection with selectedImages
    if (isModalOpen) {
      setModalSelection(selectedImages);
    }
  }, [isModalOpen, selectedImages]);

  useEffect(() => {
    if (experimentData) {
      setSelectedAlgorithms(experimentData.algorithms);
      setSelectedImages(experimentData.data.map(image => image.name));
    }
  }, [experimentData]);


  // Add logic to handle image selection, algorithm selection, etc.
  const handleImageSelectInModal = (image: string) => {
    setModalSelection(prevSelection => {
      if (prevSelection.includes(image)) {
        return prevSelection.filter(item => item !== image);
      } else {
        return [...prevSelection, image];
      }
    });
  };

  const handleConfirmSelection = () => {
    setSelectedImages(modalSelection);
    setIsModalOpen(false);
  };

  const handleCancelSelection = (image: string) => {
    setSelectedImages(prevSelection => prevSelection.filter(item => item !== image));
  };

  const addAlgorithm = (algorithm: string) => {
    if (!selectedAlgorithms.includes(algorithm)) {
      setSelectedAlgorithms([...selectedAlgorithms, algorithm]);
    }
  };

  const removeAlgorithm = (algorithm: string) => {
    setSelectedAlgorithms(selectedAlgorithms.filter(alg => alg !== algorithm));
  };


  const handleRunExperiment = () => {
    if (selectedImages.length === 0 || selectedAlgorithms.length === 0) {
      setExperimentOutput('Please select at least one image and one algorithm');
      return;
    }

    setLoading(true); // Show loading modal

    setTimeout(() => {
      setLoading(false); // Hide loading modal after 1 second
      setPredictions(makePredictions(selectedImages));
      setEvaluations(makeEvaluations(selectedImages, selectedAlgorithms));
      setIsExperimentRun(true);
    }, 3000);

  };

  const handleImageClick = (imageName: string) => {
    setSelectedImages(prevSelectedImages => {
      return prevSelectedImages.includes(imageName)
        ? prevSelectedImages.filter(name => name !== imageName)
        : [...prevSelectedImages, imageName];
    });
  };

  


  return (
    <Box sx={{ mt: 3, mb: 3, ml: 1, pb: 3, borderBottom: 1, minHeight: "600px" }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={3}>
          {/* Sidebar */}
          <Box sx={{ borderRight: 1, borderColor: 'divider', m: 2 }}>
            {/* Images Box */}
            <Box sx={{ mb: 3, borderBottom: 1, borderColor: 'divider', padding: 2 }}>
              <Typography variant="h6">Images</Typography>
              <Button variant="contained" color="primary" onClick={() => setIsModalOpen(true)} sx={{ mt: 2 }}>Select Images</Button>
              <Box sx={{ mt: 3 }}>
                {selectedImages.map((image, index) => (
                  <Chip key={index} label={image} onDelete={() => handleCancelSelection(image)} sx={{ mt: 1 }} />
                ))}
              </Box>
            </Box>
            {/* Algorithms Box */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Algorithms</Typography>
              <Box sx={{ mt: 3, border: 1, borderColor: 'divider', padding: 1 }}>
                {selectedAlgorithms.map((algorithm, index) => (
                  <Chip 
                    key={index} 
                    label={algorithm}
                    onDelete={() => removeAlgorithm(algorithm)} 
                    deleteIcon={<CloseIcon />}
                    sx={{ m: 0.5 }}
                  />
                ))}
              </Box>
              <Box sx={{ mt: 2 }}>
                {experimentData?.algorithms.map((algorithm, index) => (
                  !selectedAlgorithms.includes(algorithm) && (
                    <Chip 
                      key={index} 
                      label={algorithm} 
                      onClick={() => addAlgorithm(algorithm)}
                      sx={{ m: 0.5 }}
                    />
                  )
                ))}
              </Box>
            </Box>
          </Box>
        </Grid>
        <Grid item xs={12} md={9}>
          {/* Experiment Visualization */}
          <Box sx={{ pl: 2 }}>
            <Typography variant="h5">{experimentData?.name}</Typography>
            <Button variant="contained" color="secondary" onClick={handleRunExperiment} sx={{ mt: 2 }}>Run Experiment</Button>
            {/* Experiment Visualization */}
            {isExperimentRun && (
              <Visualizations 
                experimentData={experimentData}
                predictions={predictions}
                evaluations={evaluations}
                selectedImages={selectedImages}
                selectedAlgorithms={selectedAlgorithms}
              />
            )}
          </Box>
        </Grid>
      </Grid>
      
      {/* Image Selection Dialog */}
      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Select Images</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {experimentData?.data.map((imageData, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Paper 
                  sx={{ 
                    height: "200px", 
                    display: 'flex', 
                    justifyContent: 'center', 
                    alignItems: 'center', 
                    cursor: 'pointer', 
                    opacity: selectedImages.includes(imageData.name) ? 0.5 : 1
                  }}
                  onClick={() => handleImageClick(imageData.name)}
                >
                  <Plot data={JSON.parse(imageData.json).data} layout={JSON.parse(imageData.json).layout} />
                  <Typography variant="subtitle1" align="center">{imageData.name}</Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsModalOpen(false)}>Cancel</Button>
          <Button onClick={handleConfirmSelection}>OK</Button>
        </DialogActions>
      </Dialog>

      {/* Loading Dialog */}
      <Dialog open={loading}>
        <DialogContent sx={{ textAlign: 'center' }}>
          <CircularProgress />
          <Typography variant="h6" sx={{ mt: 2 }}>Loading</Typography>
        </DialogContent>
      </Dialog>
    </Box>
  );
}

export default ExperimentComponent;




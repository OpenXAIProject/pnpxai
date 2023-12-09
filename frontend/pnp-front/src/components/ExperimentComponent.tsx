// src/components/ExperimentComponent.tsx
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
import { Experiment, InputData } from '../app/types';
import { fetchExperiment } from '../features/apiService';

const ExperimentComponent: React.FC<{experiment: Experiment, key: number}> = ( {experiment} ) => {
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

  const inputs: InputData[] = JSON.parse(JSON.stringify(experiment.inputs));
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [modalSelection, setModalSelection] = useState<string[]>([]); // State to track selection in the modal
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [experimentOutput, setExperimentOutput] = useState<string>('');
  const [experimentResult, setExperimentResult] = useState<any[]>([]);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [evaluations, setEvaluations] = useState<any[]>([]);
  const [isExperimentRun, setIsExperimentRun] = useState(false); // New state to track if experiment is run
  const [loading, setLoading] = useState(false); // State for loading modal

  
  useEffect(() => {
    setSelectedAlgorithms(experiment.explainers.map(explainer => explainer.name));
    // console.log(selectedAlgorithms);
    // console.log(experiment.explainers)
  }, []);

  useEffect(() => {
    // Synchronize modalSelection with selectedImages
    if (isModalOpen) {
      setModalSelection(selectedImages);
    }
  }, [isModalOpen, selectedImages]);


  const modifyLayout = (layout: any) => {
      const modifiedLayout = {
          ...layout,
          xaxis: { visible: false },
          yaxis: { visible: false },
          width: 240,
          height: 200,
          margin: { l: 0, r: 0, t: 0, b: 0 }
      };

      modifiedLayout.template.data.heatmap[0].colorbar = null;

      return modifiedLayout;
  }

  const modifyData = (data: any) => {
      const modifiedData = {
        ...data[0],
        coloraxis: null,
        showscale: false,
      }

      return [modifiedData];
  }

  const preprocess = (response: any) => {
      response.data.data.forEach((result: any) => {
          result.input = JSON.parse(result.input);
          result.input.layout = modifyLayout(result.input.layout);

          result.visualizations.forEach((visualization: any) => {
              visualization.data = JSON.parse(visualization.data);
              visualization.data.data = modifyData(visualization.data.data);
              visualization.data.layout = modifyLayout(visualization.data.layout);
          });
      });
  }



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


  const handleRunExperiment = async () => {
    if (selectedImages.length === 0 || selectedAlgorithms.length === 0) {
      setExperimentOutput('Please select at least one image and one algorithm');
      return;
    }
    
    setLoading(true); // Show loading modal

    // Convert the argument for API
    const selectedExplainers = selectedAlgorithms.map(algorithm => {
      const obj = experiment.explainers.find(explainer => explainer.name === algorithm);
      return obj?.id;
    }).filter(id => id !== null);

    const selectedInputs = selectedImages.map(image => Number(image));

    const projectId = "test_project"; // Replace with your actual project ID
    const response = await fetchExperiment(projectId, experiment.name, {
      inputs: selectedInputs,
      explainers: selectedExplainers,
    });
    
    preprocess(response);
    setExperimentResult(response.data.data);
    setLoading(false); // Hide loading modal after 1 second
    const preds = makePredictions(selectedImages);
    preds[0].isCorrect = false; // For testing
    const evals = makeEvaluations(selectedImages, selectedAlgorithms);
    setPredictions(preds);
    setEvaluations(evals);
    setIsExperimentRun(true);

    


    // setTimeout(() => {
    //   setLoading(false); // Hide loading modal after 1 second
    //   const preds = makePredictions(selectedImages);
    //   preds[0].isCorrect = false; // For testing
    //   const evals = makeEvaluations(selectedImages, selectedAlgorithms);
    //   setPredictions(preds);
    //   setEvaluations(evals);
    //   setIsExperimentRun(true);
    // }, 500);

    // console.log(JSON.parse(response.data.data[0].input));
    // console.log(response.data.data[0].visualizations[0]);
    // console.log(JSON.parse(response.data.data[0].visualizations[0].data));

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
                {experiment?.explainers.map(explainer => explainer.name).map((algorithm, index) => (
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
            <Typography variant="h5">{experiment?.name}</Typography>
            <Button variant="contained" color="secondary" onClick={handleRunExperiment} sx={{ mt: 2 }}>Run Experiment</Button>
            {/* Experiment Visualization */}
            {isExperimentRun && experimentResult && experimentResult.length > 0 && (
              <Visualizations 
                experimentResult={experimentResult}
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
            {inputs.map((input, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Paper 
                  sx={{ 
                    height: "300px", 
                    display: 'flex', 
                    flexDirection: 'column', // Set the flex direction to column
                    justifyContent: 'center', // Aligns children vertically in the center
                    alignItems: 'center', // Aligns children horizontally in the center
                    cursor: 'pointer', 
                    opacity: selectedImages.includes(input.id) ? 0.5 : 1
                  }}
                  onClick={() => handleImageClick(input.id)}
                >
                  <Paper
                    sx={{
                      height: "300px",
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      alignItems: 'center',
                      cursor: 'pointer',
                      opacity: selectedImages.includes(input.id) ? 0.5 : 1
                    }}
                    onClick={() => handleImageClick(input.id)}
                  >
                  </Paper>
                    {/* <Plot data={input.imageObj.data} layout={input.imageObj.layout} /> */}
                    <img src={input.imageObj.data[0].source} width={240} height={200} alt={input.id} />
                    <Typography variant="subtitle1" align="center">{input.id}</Typography>
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




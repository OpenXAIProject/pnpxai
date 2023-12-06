// src/pages/AutoExplainPage.tsx
import React, { useState, useEffect } from 'react';
import { Grid, Container, Tabs, Tab, Box, Button, Typography, Card, CardContent } from '@mui/material';
import Sidebar from '../components/SideBar/SideBar';
import ImageClassificationResults from '../components/ImageClassificationResult';
import mockResult from '../assets/mockup/mockResults.json';

interface ModelPrediction {
  label: string;
  probability: number;
}

interface ImageClassificationResult {
  imageName : string;
  imagePath: string;
  trueLabel: string;
  modelPredictions: ModelPrediction[];
  isCorrect: boolean;
}

const mockResults: ImageClassificationResult[] = mockResult;


const experiments = [
  {
    name: "Experiment 1",
    algorithms: ["Algorithm 1", "Algorithm 2", "Algorithm 3"],
    images : mockResult.slice(0,3)
  },
  {
    name: "Experiment 2",
    algorithms: ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4"],
    images : mockResult.slice(3,6)
    
  },
];

const ModelExplainPage: React.FC = () => {
  const [selectedExperimentTab, setSelectedExperimentTab] = useState(0);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<Record<number, string[]>>({});
  const [numObjectsinLine, setNumObjectsinLine] = useState(3); // Default value

  useEffect(() => {
    const initialSelectedAlgorithms = experiments.reduce((acc, _, index) => {
      acc[index] = experiments[index].algorithms;
      return acc;
    }, {} as Record<number, string[]>);
    setSelectedAlgorithms(initialSelectedAlgorithms);
  }, []);

  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [filteredResults, setFilteredResults] = useState<ImageClassificationResult[]>([]);

  const handleRunClick = () => {
    setFilteredResults(mockResults.filter(result => selectedImages.includes(result.imageName)));
  };

  const handleImageSelectionChange = (selectedImages: string[]) => {
    setSelectedImages(selectedImages);
  };

  return (
    <Grid container direction="row" spacing={0} sx={{ flexGrow: 1, width: '100%', minHeight: 800 }}>
      {/* Sidebar Grid */}
      <Grid item xs={3}>
        <Sidebar
          experiments={experiments}
          selectedExperimentTab={selectedExperimentTab}
          selectedAlgorithms={selectedAlgorithms}
          drawerWidth={300}
          setSelectedAlgorithms={setSelectedAlgorithms}
          onImageSelectionChange={handleImageSelectionChange}
          setNumObjectsinLine={setNumObjectsinLine}
        />
      </Grid>

      {/* Main Content Grid */}
      <Grid item xs={9} sx={{ marginTop: 3 }}>
        <Container maxWidth="lg">
          {/* Instructions Card */}
          {/* ... Instructions Card Content ... */}

          {/* Experiment Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={selectedExperimentTab} onChange={(event, newValue) => setSelectedExperimentTab(newValue)} aria-label="Experiment tabs">
              {experiments.map((value, index) => (
                <Tab label={value['name']} key={index} />
              ))}
            </Tabs>
          </Box>

          {/* Auto Explain Button */}
          <Button variant="contained" color="primary" onClick={handleRunClick}>
            Auto Explain
          </Button>

          {/* Image Classification Results */}
          {filteredResults.length > 0 && (
            <Box sx={{ mt: 2, width: 1000 }}>
              <Typography variant='h2' sx={{ m: 3 }}> 아래 내용은 설명된 부분입니다 </Typography>
              <ImageClassificationResults numObjectsinLine={numObjectsinLine} algorithms={selectedAlgorithms[selectedExperimentTab]} results={filteredResults} />
            </Box>
          )}
        </Container>
      </Grid>
    </Grid>
  );
};

export default ModelExplainPage;


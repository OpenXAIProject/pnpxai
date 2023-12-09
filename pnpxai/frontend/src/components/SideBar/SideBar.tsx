// src/components/SideBar/SideBar.tsx
import React, { useState } from 'react';
import { 
  Paper, List, ListItem, FormControl, 
  InputLabel, Select, MenuItem, Divider, 
  Chip, Box, Button, Container} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Remove';
import ImageModal from '../Modal/imageModal';

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

interface SidebarProps {
    experiments: {
        name: string;
        algorithms: string[];
        images : ImageClassificationResult[]
    }[];
    selectedExperimentTab: number;
    selectedAlgorithms?: Record<number, string[]>;
    drawerWidth?: number;
    setSelectedAlgorithms?: (selectedAlgorithms: Record<number, string[]>) => void;
    setNumObjectsinLine?: (numObjects: number) => void;
    onImageSelectionChange: (selectedImages: string[]) => void; // New prop
}

const Sidebar: React.FC<SidebarProps> = ({ 
  experiments, 
  selectedExperimentTab, 
  selectedAlgorithms, 
  drawerWidth,
  setSelectedAlgorithms, 
  setNumObjectsinLine,
  onImageSelectionChange,
}) => {
  const ListItemMargin = 4;
  
  const images = experiments[selectedExperimentTab].images;
  const maxItems = experiments[selectedExperimentTab].algorithms.length;
  const [imageSelection, setImageSelection] = useState<ImageClassificationResult[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [numObjects, setNumObjects] = useState(maxItems);

  

  const availableAlgorithms = experiments[selectedExperimentTab].algorithms.filter(
    algo => !selectedAlgorithms?.[selectedExperimentTab]?.includes(algo)
  );


  const handleAlgorithmSelect = (algorithm: string) => {
    const updatedSelectedAlgorithms = {
      ...selectedAlgorithms,
      [selectedExperimentTab]: [...(selectedAlgorithms?.[selectedExperimentTab] || []), algorithm],
    };
    setSelectedAlgorithms && setSelectedAlgorithms(updatedSelectedAlgorithms);
  };

  const handleDelete = (algorithm: string) => {
    const updatedSelectedAlgorithms = {
      ...selectedAlgorithms,
      [selectedExperimentTab]: selectedAlgorithms?.[selectedExperimentTab]?.filter(a => a !== algorithm) ?? [],
    };
    setSelectedAlgorithms && setSelectedAlgorithms(updatedSelectedAlgorithms);
  };

  const handleSelectImage = (imageName: string) => {
    setImageSelection(prev => {
      const newSelection = prev.some(img => img.imageName === imageName) 
        ? prev.filter(img => img.imageName !== imageName) 
        : [...prev, images.find(img => img.imageName === imageName)!]; // Safely assuming the image will be found
      onImageSelectionChange(newSelection.map(img => img.imageName)); // Update parent component
      return newSelection;
    });
  };

  const handleSaveSelectedImages = (selectedImages: ImageClassificationResult[]) => {
    setImageSelection(selectedImages);
    onImageSelectionChange(selectedImages.map(img => img.imageName));
  }

  return (
    <Paper
      sx={{
        width: drawerWidth,
        height: '100%',
        boxSizing: 'border-box',
      }}
    >
      <List>
        {/* Image Selection */}
        <ListItem sx={{ mb: ListItemMargin }}>
          <FormControl fullWidth>
            <Button 
              onClick={() => setIsModalOpen(true)}
              sx={{
                border: '1px solid #a0a0a0', // Example border styling
                ':hover': {
                  border: '1px solid #a0a0a0', // Optional: change border on hover
                }}}
            >
              Select Images
            </Button>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
              {imageSelection.map((image, index) => (
                <Chip key={index} label={image.imageName} onDelete={() => handleSelectImage(image.imageName)} />
              ))}
            </div>
          </FormControl>
        </ListItem>

        <Divider />

        {/* Algorithms */}
        <ListItem sx={{ mb : ListItemMargin}}>
          <FormControl fullWidth>
            <InputLabel id="algorithm-label">Algorithms</InputLabel>
            <Select
              labelId="algorithm-label"
              id="algorithm"
              value=""
              label="Algorithms"
              onChange={(e) => handleAlgorithmSelect(e.target.value)}
              displayEmpty
            >
              {availableAlgorithms.map((algorithm, index) => (
                  <MenuItem key={index} value={algorithm}>{algorithm}</MenuItem>))
              }
            </Select>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: '10px', margin: '10px 0' }}>
              {selectedAlgorithms?.[selectedExperimentTab]?.map((algorithm, index) => (
                <Chip key={index} label={algorithm} onDelete={() => handleDelete(algorithm)} deleteIcon={<DeleteIcon />} />))
              }
            </Box>
          </FormControl>
        </ListItem>

        <Divider />

        {/* Number of Objects in Array */}
        <ListItem sx={{ mb : ListItemMargin}}>
          <FormControl fullWidth>
            <InputLabel id="num-objects-label">Num of Object in Array</InputLabel>
            <Select
              labelId="num-objects-label"
              id="num-objects"
              value={numObjects}
              label="Num of Object in Array"
              onChange={(e) => {
                setNumObjects(Number(e.target.value))
                setNumObjectsinLine && setNumObjectsinLine(Number(e.target.value))
              }}
            >
              {/* Range from max items */}
              {Array.from(Array(maxItems).keys()).map((num, index) => (
                <MenuItem key={index} value={num+1}>{num+1}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </ListItem>
      </List>
      <ImageModal
        open={isModalOpen}
        images={images} // Directly passing images from experiments
        selectedImages={imageSelection}
        onClose={() => setIsModalOpen(false)}
        onSelect={handleSelectImage}
        onSave={handleSaveSelectedImages}
      />
    </Paper>
  );
};

export default Sidebar;
// src/components/Modal/imageModal.tsx
import React, { useState, useEffect } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Grid, Paper, Typography, Button } from '@mui/material';

interface Image {
  name: string;
  src: string;
}

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

interface ImageModalProps {
  open: boolean;
  images: ImageClassificationResult[];  // Updated type
  selectedImages: ImageClassificationResult[]; 
  onClose: () => void;
  onSelect: (imageName: string) => void;
  onSave: (selectedImages: ImageClassificationResult[]) => void;
}



const ImageModal: React.FC<ImageModalProps> = ({ 
  open, 
  images, 
  selectedImages, 
  onClose, 
  onSelect,
  onSave,
}) => {
  const [localSelectedImages, setLocalSelectedImages] = useState<ImageClassificationResult[]>([]);

  useEffect(() => {
    if (open) {
      setLocalSelectedImages(selectedImages);
    }
  }, [open, selectedImages]);

  const handleLocalSelect = (imageName: string) => {
    const found = localSelectedImages.find(img => img.imageName === imageName);
    if (found) {
      setLocalSelectedImages(localSelectedImages.filter(img => img.imageName !== imageName));
    } else {
      const newImage = images.find(img => img.imageName === imageName);
      if (newImage) {
        setLocalSelectedImages([...localSelectedImages, newImage]);
      }
    }
  };

  const handleCancel = () => {
    onClose();
  };

  const handleOk = () => {
    onSave(localSelectedImages);
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleCancel} fullWidth maxWidth="md">
      <DialogTitle>Select Images</DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          {images.map((image, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper
                sx={{ cursor: 'pointer', opacity: localSelectedImages.includes(image) ? 0.5 : 1 }}
                onClick={() => handleLocalSelect(image.imageName)}
              >
                <img src={image.imagePath} alt={image.imageName} style={{ width: '240px', height: '200px' }} />
                <Typography variant="subtitle1" align="center">{image.imageName}</Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCancel} color="primary">
          Cancel
        </Button>
        <Button onClick={handleOk} color="primary">
          OK
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ImageModal;

import React, { useState } from 'react';
import { InputData } from '../../app/types';
import { Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, Grid, Paper, Typography } from '@mui/material';

// Define the props interface
interface GalleryModalProps {
  galleryInputs: InputData[];
  isModalOpen: boolean;
  setIsModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setInputs: React.Dispatch<React.SetStateAction<string[]>>;
}

const GalleryModal: React.FC<GalleryModalProps> = ({ 
  galleryInputs, 
  isModalOpen,
  setIsModalOpen,
  setInputs,
}) => {
  
  const [tmpInputs, setTmpInputs] = useState<string[]>([]);

  const handleImageClick = (imageId: string) => {
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


  return (
    <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} fullWidth maxWidth="md">
      <DialogTitle>Select Intances</DialogTitle>
        <DialogContent>
        {galleryInputs.length > 0 ? (
        <Grid container spacing={2}>
          {galleryInputs.map((input, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper 
                sx={{ 
                  height: "300px", 
                  display: 'flex', 
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
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
        ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <CircularProgress />
        </Box>
        )
        }
        </DialogContent>
      <DialogActions>
        <Button onClick={handleCancelSelection}>Cancel</Button>
        <Button onClick={handleConfirmSelection}>OK</Button>
      </DialogActions>
    </Dialog>
  )
};

export default GalleryModal;
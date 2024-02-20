import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../app/store';
import {
  setGalleryInputs,
  setInputs,
} from '../../features/globalState';
import { Experiment, InputData } from '../../app/types';
import { Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, Grid, Paper, Typography } from '@mui/material';
import { ErrorProps, ErrorSnackbar } from './ErrorSnackBar';
import { fetchInputsByExperimentId } from '../../features/apiService';

// Define the props interface
interface GalleryModalProps {
  experiment: Experiment;
  galleryInputs: InputData[];
  isModalOpen: boolean;
  setIsModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

const GalleryModal: React.FC<GalleryModalProps> = ({ 
  experiment,
  isModalOpen,
  setIsModalOpen,
}) => {

  const dispatch = useDispatch();
  const projectId = useSelector((state: RootState) => state.global.status.currentProject);
  const expId = experiment.id;
  const expCache = useSelector((state: RootState) => state.global.expCache.filter((item) => item.projectId === projectId && item.expId === expId)[0]);
  const galleryInputs = expCache?.galleryInputs

  const [isError, setIsError] = useState(false);
  const [errorInfo, setErrorInfo] = useState<ErrorProps[]>([]);
  
  const [tmpInputs, setTmpInputs] = useState<InputData[]>([]);
  const handleImageClick = (input: InputData) => {
    if (tmpInputs.map(item => item.id).includes(input.id)) {
      setTmpInputs(prevItems => prevItems.filter(item => item !== input));
    } else{
      // Constrain the number of selected inputs to 1
      // If you want to remove the constraint, just remove this
      if (tmpInputs.length >= 1) {
        setTmpInputs([]);
      }
      setTmpInputs(prevItems => [...prevItems, input]);
    }
  };

  const handleConfirmSelection = () => {
    dispatch(setInputs({ projectId: projectId, expId: expId, inputs: tmpInputs }));
    setIsModalOpen(false);
  };

  const handleCancelSelection = () => {
    setIsModalOpen(false);
  };

  useEffect(() => {

    const fetchAndSetGalleryInputs = async () => {
      try {
        const response = await fetchInputsByExperimentId(projectId, expId);
        const galleryInputsBuffer = response.data.data.map((input: string, index: number) => {
          const parsedInput = JSON.parse(input);
          return {
            id: index.toString(),
            source: parsedInput.data[0].source,
          };
        });
        
        dispatch(setGalleryInputs({ projectId: projectId, expId: expId, galleryInputs: galleryInputsBuffer }));
      } catch (error: any) {
        console.error(error);
        setIsError(true);
        setErrorInfo(error.response.data.errors);
      }
    };

    if (isModalOpen && galleryInputs.length === 0) {
      fetchAndSetGalleryInputs();
    }
  }, [isModalOpen, projectId, expId, dispatch]);

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
                  opacity: tmpInputs.map(item => item.id).includes(input.id) ? 0.5 : 1
                }}
                onClick={() => handleImageClick(input)}
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
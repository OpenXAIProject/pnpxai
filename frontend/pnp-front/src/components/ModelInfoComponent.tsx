import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  Typography, Card, CardContent, CardHeader, Box, Alert,
  FormControlLabel, Toolbar, Collapse, Button } from '@mui/material';
import { RootState } from '../app/store'; // Import your RootState type

const ModelInfoComponent: React.FC = () => {
  const experimentData = useSelector((state: RootState) => {
    return state.experiments.data;
  });
  const [expanded, setExpanded] = useState<{ [key: string]: boolean }>({});
  



  const handleCollapse = (index: string | number) => {
    setExpanded(prevExpanded => ({
      ...prevExpanded,
      [index]: !prevExpanded[index]
    }));
  };
  

  const isNoModelDetected = experimentData.every((experiment) => {
    // return !experiment.modelDetected;
    return true; // For testing

  });

  return (
    <Box sx={{ m: 1 }}>
      <Typography variant='h1'> Model Detection Result </Typography>
      <Box sx={{ mt: 4 }}>
        {isNoModelDetected ? (
          <Box sx={{ m: 5, minHeight: "50px" }}>
            <Card>
              <CardContent>
                <Alert severity="warning">No available experiment. Try Again.</Alert>
              </CardContent>
            </Card>
          </Box>
        ) : (
          experimentData.map((experiment, index) => {
            const toolbarStyle = {
              backgroundColor: experiment.modelDetected ? 'green' : 'red',
              color: 'white',
            };
            return (
              <Box key={index} sx={{ m: 1 }}>
                <Card>
                  <CardHeader title={experiment.name} />
                  <Toolbar style={toolbarStyle}>
                    <Typography variant='h6'>
                      {experiment.modelDetected ? 'Model Detected' : 'Model Not Detected'}
                    </Typography>
                  </Toolbar>
                  <CardContent>
                    <Typography variant='body1'> Model: {experiment.model} </Typography>
                    <Typography variant='body1'> Availalbe XAI Algorithms: {experiment.algorithms.join(', ')} </Typography>
                    {experiment.modelDetected ? (
                      <Box sx={{ m: 1 }}>
                        <Button onClick={() => handleCollapse(index)}> View Model </Button>
                        <Collapse in={expanded[index]} timeout="auto" unmountOnExit>
                          <Typography variant='body1'> Model Structure: {experiment.modelStructure} </Typography>
                        </Collapse>
                      </Box>
                    ) : (
                      <Box sx={{ m: 5, minHeight: "50px" }}>
                        <Card>
                          <CardContent>
                            <Alert severity="warning"> Retry. Only torch.nn.Module Based model can be detected.</Alert>
                          </CardContent>
                        </Card>
                      </Box>
                    )
                  }
                    
                  </CardContent>
                </Card>
              </Box>
            );
          })
        )}
      </Box>
    </Box>
  );
};

export default ModelInfoComponent;
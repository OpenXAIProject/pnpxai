import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../app/store';
import ModelInfoComponent from '../components/ModelInfoComponent'; // Adjust the import path as per your project structure
import { Typography, Box, Card, CardContent, Button, Alert, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';

const ModelInfoPage: React.FC = () => {
  const loaded = useSelector((state: RootState) => state.projects.loaded);
  const projectsData = useSelector((state: RootState) => state.projects.data);
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const projectData = projectsData?.find(project => project.id === projectId);


  const isNoModelDetected = projectData?.experiments.every((experiment) => {
    return !experiment.id;
  });

  if (!loaded || !projectsData) {
    return (
      <Box sx={{ p: 2, maxWidth: 1400  }}>
        <Box sx={{ m: 5, minHeight: "50px" }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                <CircularProgress />
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    )
  }

  return (
    <Box sx={{ p: 2, maxWidth: 1400  }}>
      <Box>
        <Typography variant='h1'>
          Plug and Play XAI Introduction
        </Typography>            
        <Box sx={{ m : 2}}>
          <Typography variant="body2">
            PnP XAI is easy to use XAI framework AI developers
          </Typography>
        </Box>
      </Box>

      <Typography variant='h1'> Model Architecture Detection Results </Typography>
      {!isNoModelDetected ? (
        projectData?.experiments.map((experiment, index) => {
          return <ModelInfoComponent key={index} experiment={experiment} showModel={index === 0 ? true : false}/>;
        })
      ) : (
        <Box sx={{ m: 5, minHeight: "50px" }}>
          <Card>
            <CardContent>
              <Alert severity="warning">No available experiment. Try Again.</Alert>
            </CardContent>
          </Card>
        </Box>
      )}

      <Box sx={{ mt : 1, textAlign : "right"}}>
        <Typography sx={{ mb: 1 }}>
          설명 알고리즘 적용하기
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          component={Link} 
          to="/model-explanation"
        >
          Model Explanation
        </Button>
      </Box>
      
    </Box>
  );
};

export default ModelInfoPage;



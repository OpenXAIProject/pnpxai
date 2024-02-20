import React from 'react';
import { useSelector } from 'react-redux';
import { Box, CardContent, Card, Alert, CircularProgress, Typography} from '@mui/material';
import { RootState } from '../app/store';
import ExperimentComponent from '../components/ExperimentComponent';

const ExperimentPage: React.FC = () => {
  const  {loaded, error} = useSelector((state: RootState) => state.global);
  const projects = useSelector((state: RootState) => state.global.projects);
  const projectId = useSelector((state: RootState) => state.global.status.currentProject);
  const projectData = projects?.find(project => project.id === projectId);
  const isAnyModelDetected = projectData?.experiments.some(experiment => experiment.id);

  if (!loaded || projects === undefined) {
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
    <Box sx={{ p: 2 }}>
      <Typography variant='h1'> Local Explanation </Typography>
      <Box sx={{ m : 1}}>
        <Typography variant='h5'> Visualize the Explainers </Typography>
      </Box>
      <Box sx={{ m: 1 }}>
        {isAnyModelDetected ? (
          projectData?.experiments.filter(experiment => experiment.modelDetected).map((experiment, index) => (
            <ExperimentComponent key={`${projectId}-${experiment.id}`} experiment={experiment} />
          ))
        ) : (
          <Box sx={{ m: 5, minHeight: "200px", maxWidth : "350px" }}>
            <Alert severity="warning">No available experiment. Try Again.</Alert>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ExperimentPage;

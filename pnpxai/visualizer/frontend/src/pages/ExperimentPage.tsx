import React from 'react';
import { useSelector } from 'react-redux';
import { Box, CardContent, Card, Alert, CircularProgress
} from '@mui/material';
import { RootState } from '../app/store';
import ExperimentComponent from '../components/ExperimentComponent';

const ExperimentPage: React.FC = () => {
  const loaded = useSelector((state: RootState) => state.projects.loaded);
  const projectsData = useSelector((state: RootState) => state.projects.data);
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const projectData = projectsData?.find(project => project.id === projectId);
  const isAnyModelDetected = projectData?.experiments.some(experiment => experiment.id);

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
    <Box sx={{ m: 1 }}>
      <Box sx={{ m: 1 }}>
        {isAnyModelDetected ? (
          projectData?.experiments.filter(experiment => experiment.modelDetected).map((experiment, index) => (
            <ExperimentComponent key={index} experiment={experiment} />
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

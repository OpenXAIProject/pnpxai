import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { Box, Container, CardContent, Typography, Card, Alert, 
  FormControl, InputLabel, Select, MenuItem, Button, Collapse, CircularProgress
} from '@mui/material';
import { RootState } from '../app/store';
import ExperimentComponent from '../components/ExperimentComponent';

const ExperimentPage = () => {
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
      {/* <Box sx={{ mt: 4, mb: 4}}>
        <Container>
          <Card>
            <CardContent>
              <Typography> 이 곳에서 자동으로 모델 설명을 보실 수 있습니다. </Typography>
              <Typography> 각각의 Experiment에서 원하는 샘플 데이터, 원하는 설명 알고리즘을 선택하면 자동으로 선택해드립니다.</Typography>
              <Typography> 자동으로 선택된 데이터와 알고리즘을 바탕으로 설명 결과를 보실 수 있습니다.</Typography>
              <Button onClick={() => handleCollapse()}> Output format details </Button>
              <Collapse in={expanded} timeout="auto" unmountOnExit>
                <Typography> 1. Label : 실제 이미지의 분류 </Typography>
                <Typography> 2. Prediction Probabilities : 모델의 예측한 분류 확률 </Typography>
                <Typography> 3. IsCorrect : 모델이 정확하게 예측했는지 측정 </Typography>
                <Typography> 평가 지표 </Typography>
                <Typography> 1. Faithfulness : 설명 결과가 실제 모델의 동작을 잘 반영하는지 측정 </Typography>
                <Typography> 2. Robustness : 설명 결과가 모델의 동작에 얼마나 영향을 받는지 측정 </Typography>
              </Collapse>
            </CardContent>
          </Card>
        </Container>
      </Box> */}
      

      
      {/* <Box sx={{ ml : 40}}>
      <Typography variant='h1'> Local Explanation </Typography>
      <Typography variant='h6'> You can visualize the results </Typography>

      </Box> */}


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

import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Box, Container, CardContent, Typography, Card, Alert } from '@mui/material';
import { RootState } from '../app/store';
import ExperimentComponent from '../components/ExperimentComponent';
import { fetchData } from '../features/experimentSlice';
import { AnyAction } from 'redux';
import { ThunkDispatch } from 'redux-thunk';

const ExperimentPage = () => {
  const experimentData = useSelector((state: RootState) => state.experiments.data);
  const dispatch = useDispatch<ThunkDispatch<RootState, null, AnyAction>>();

  useEffect(() => {
    dispatch(fetchData());
  }, [dispatch]);

  const isAnyModelDetected = experimentData.some(experiment => experiment.modelDetected);
  // const isAnyModelDetected = false; // For testing

  return (
    <Box sx={{ m: 1 }}>
      <Box sx={{ mt: 4, mb: 4}}>
        <Container>
          <Card>
            <CardContent>
              <Typography> 이 곳에서 자동으로 모델 설명을 보실 수 있습니다. </Typography>
              <Typography> 각각의 Experiment에서 원하는 샘플 데이터, 원하는 설명 알고리즘을 선택하면 자동으로 선택해드립니다.</Typography>
              <Typography> 자동으로 선택된 데이터와 알고리즘을 바탕으로 설명 결과를 보실 수 있습니다.</Typography>
              <Typography> 설명 결과는 다음과 같이 나타납니다.</Typography>
              <Typography> 1. Label : 실제 이미지의 분류 </Typography>
              <Typography> 2. Prediction Probabilities : 모델의 예측한 분류 확률 </Typography>
              <Typography> 3. IsCorrect : 모델이 정확하게 예측했는지 측정 </Typography>
              <Typography> 평가 지표 </Typography>
              <Typography> 1. Faithfulness : 설명 결과가 실제 모델의 동작을 잘 반영하는지 측정 </Typography>
              <Typography> 2. Robustness : 설명 결과가 모델의 동작에 얼마나 영향을 받는지 측정 </Typography>
            </CardContent>
          </Card>
        </Container>
      </Box>

      <Box sx={{ m: 1 }}>
        {isAnyModelDetected ? (
          experimentData.filter(experiment => experiment.modelDetected).map((experiment, index) => (
            <ExperimentComponent key={index} id={experiment.experiment_id} />
          ))
        ) : (
          <Box sx={{ m: 5, minHeight: "200px" }}>
            <Card>
              <CardContent>
                <Alert severity="warning">No available experiment. Try Again.</Alert>
              </CardContent>
            </Card>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ExperimentPage;

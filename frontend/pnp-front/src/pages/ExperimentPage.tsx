import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Box, CardContent, Typography, Card, Alert } from '@mui/material';
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

  return (
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
  );
};

export default ExperimentPage;

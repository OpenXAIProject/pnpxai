// src/pages/ModelExplainPage.tsx
import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Container, Grid, Card, Select, MenuItem, Button, Typography, Chip } from '@mui/material';
import { fetchData, selectData, DataType } from '../features/yourDataSlice';
import { selectAlgorithms, AlgorithmType } from '../features/algorithmSlice';
import { RootState } from '../app/store'; // Adjust this import path as necessary
import { setSelectedData, setSelectedAlgorithms, setPlotData, setLabelData } from '../features/explainSlice';
import ModelExplanation from '../components/ModelExplanation/ModelExplanation';


const mockLabelData = [
  { id: 1, realLabel: 'Cat', predictedLabel: 'Cat', evaluationResults: { accuracy: 0.95, precision: 0.93 } },
  { id: 2, realLabel: 'Dog', predictedLabel: 'Dog', evaluationResults: { accuracy: 0.90, precision: 0.89 } },
  { id: 3, realLabel: 'Bird', predictedLabel: 'Cat', evaluationResults: { accuracy: 0.75, precision: 0.80 } },
  // Add more data as needed
];

const AutoExplainPage: React.FC = () => {
  const dispatch = useDispatch();
  const data: DataType[] = useSelector(selectData);
  const algorithms: AlgorithmType[] = useSelector(selectAlgorithms);
  const selectedData = useSelector((state: RootState) => state.explain.selectedData);
  const selectedAlgorithms = useSelector((state: RootState) => state.explain.selectedAlgorithms);
  const [showPlot, setShowPlot] = useState(false); // Add state to control plot visibility

  useEffect(() => {
      dispatch(fetchData());
      dispatch(setLabelData([
          ...mockLabelData
      ]));
  }, [dispatch]);

  const handleDataChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    const newData = event.target.value as number;
    if (!selectedData.includes(newData)) {
      dispatch(setSelectedData([...selectedData, newData]));
    }
  };

  const handleAlgorithmChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    const newAlgorithm = event.target.value as number;
    if (!selectedAlgorithms.includes(newAlgorithm)) {
      dispatch(setSelectedAlgorithms([...selectedAlgorithms, newAlgorithm]));
    }
  };

  const handleDataDelete = (dataId: number) => {
    dispatch(setSelectedData(selectedData.filter(id => id !== dataId)));
  };

  const handleAlgorithmDelete = (algorithmId: number) => {
    dispatch(setSelectedAlgorithms(selectedAlgorithms.filter(id => id !== algorithmId)));
  };

  const runAnalysis = () => {
      // Add code to run analysis here
      // Change show plot state to true to show the plot
      setShowPlot(true);
  };

  return (
      <Grid container spacing={2} sx={{ marginTop : 2 }}>
        <Grid item xs={12}>
          <Typography variant="h4">자동으로 설명하기</Typography>
        </Grid>
        <Grid item xs={12}>
          <Button variant="contained" color="primary" onClick={runAnalysis}>
            Run
          </Button>
        </Grid>
      </Grid>
      
  );
};

export default AutoExplainPage;

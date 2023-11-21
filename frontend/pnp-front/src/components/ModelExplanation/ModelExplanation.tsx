import React from 'react';
import { useSelector } from 'react-redux';
import { Container, Grid, Card, Typography } from '@mui/material';
import { RootState } from '../../app/store';

interface ModelExplanationProps {
  selectedData: number[]; // Props for selected data
  selectedAlgorithms: number[]; // Props for selected algorithms
}

const ModelExplanation: React.FC<ModelExplanationProps> = ({ selectedData, selectedAlgorithms }) => {
  const labelData = useSelector((state: RootState) => state.explain.labelData);

  return (
    <>
      {selectedData.map(inputId => (
        <Grid item xs={12} key={inputId}>
          <Card>
            <Typography variant="h6">{`Real Label: ${labelData.find(item => item.id === inputId)?.realLabel}`}</Typography>
            <Typography variant="h6">{`Predicted Label: ${labelData.find(item => item.id === inputId)?.predictedLabel}`}</Typography>
            <Typography variant="body1">{`Accuracy: ${labelData.find(item => item.id === inputId)?.evaluationResults.accuracy}`}</Typography>
            <Typography variant="body1">{`Precision: ${labelData.find(item => item.id === inputId)?.evaluationResults.precision}`}</Typography>
          </Card>
        </Grid>
      ))}

      {/* Embed the HTML file using an iframe */}
      <Grid item xs={12}>
        <Container sx={{ margin : 2 }}>
        <iframe 
          src="/src/assets/mockup/image_and_score_plot.html" 
          style={{ width : 1200, height : 800, border: 'none' }}
          title="Plotly Chart"
          />
        </Container>
      </Grid>
    </>
  );
};

export default ModelExplanation;

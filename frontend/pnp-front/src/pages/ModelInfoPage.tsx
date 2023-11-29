import React from 'react';
import ModelInfoComponent from '../components/ModelInfoComponent'; // Adjust the import path as per your project structure
import { Typography, Box, Card, CardContent, Button } from '@mui/material';
import { Link } from 'react-router-dom';

const ModelInfoPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, maxWidth: 1400  }}>
      <Box>
        <Typography variant='h1'>
          PnP XAI Introduction
        </Typography>            
        <Box sx={{ m : 2}}>
          <Card>
            <CardContent>
              <Typography variant="h5" component="div">
                PnP XAI
              </Typography>
              <Typography variant="body2">
                PnP XAI is easy to use XAI framework AI developers
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>

      <ModelInfoComponent />

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

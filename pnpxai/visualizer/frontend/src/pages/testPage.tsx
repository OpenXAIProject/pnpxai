// src/pages/TestPage.tsx
import React, { useEffect } from 'react';
import { Container, Box, Typography, Grid, Paper } from '@mui/material';

const TestPage: React.FC = () => {

  return (
    <Container maxWidth="lg">
      <Box sx={{ m:2 }}>
        <Typography variant='h1'> Test Page </Typography>
      </Box>
    </Container>
  )
};

export default TestPage;


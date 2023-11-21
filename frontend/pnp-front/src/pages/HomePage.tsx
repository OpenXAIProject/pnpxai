// src/pages/HomePage.tsx
import React from 'react';
import { Box, Container, Typography, Paper, useTheme } from '@mui/material';

const HomePage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="60vh"
      >
        <Typography variant="h1" gutterBottom>
          Plug and Play XAI
        </Typography>
        <Typography variant="h5">
          Easy to use XAI
        </Typography>
      </Box>
    </Container>
  );
};

export default HomePage;

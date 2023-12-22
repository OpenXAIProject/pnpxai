// src/pages/NotFoundPage.tsx
import React from 'react';
import { Container, Box, Typography } from '@mui/material';

const NotFoundPage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ p: 2 }}>
        <Typography variant="h1">404 Not Found</Typography>
      </Box>
    </Container>
  )
};

export default NotFoundPage;

// src/pages/HomePage.tsx
import React from 'react';
import { Box, Container, Typography, Paper, useTheme } from '@mui/material';
import { styled } from '@mui/material/styles';
import backgroundImage from '../assets/images/background.png';

// If you want to add a custom font, you would add it in your theme or import it in your index.html

// If you have a specific image to use, replace 'graphicImageURL' with your image path

// Styled components using MUI's 'styled' API
const HeroContainer = styled(Paper)(({ theme }) => ({
  position: 'relative',
  color: theme.palette.common.white,
  backgroundImage: `url(${backgroundImage})`,
  backgroundSize: 'cover',
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'center',
  padding: theme.spacing(6, 0),
}));

const HomePage: React.FC = () => {
  const theme = useTheme();

  return (
    <Container maxWidth="lg">
      <HeroContainer>
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
      </HeroContainer>
      <Box my={4}>
        {/* Additional content goes here */}
      </Box>
    </Container>
  );
};

export default HomePage;

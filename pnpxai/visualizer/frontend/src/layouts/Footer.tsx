// src/components/Footer/Footer.tsx
import React from 'react';
import { Box, Container, Typography, Link } from '@mui/material';

const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      sx={{
        bgcolor: 'background.paper',
        py: 3,
        mt: 4,
        borderTop: 1,
        borderColor: 'divider',
        marginTop: 0,
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="body1" align="center">
          © {new Date().getFullYear()} PnP XAI
        </Typography>
        <Typography variant="body2" align="center" sx={{ mt: 1 }}>
          An open source project. Contribute on {' '}
          <Link href="https://github.com/OpenXAIProject/pnpxai" target="_blank" rel="noopener noreferrer">
            GitHub
          </Link>.
        </Typography>
      </Container>
    </Box>
  );
};

export default Footer;

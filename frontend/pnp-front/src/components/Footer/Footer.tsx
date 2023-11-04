import React from 'react';
import { Grid, Box, Link, Typography, Container } from '@mui/material';

// Define the structure for props if necessary (none needed for this simple footer)
// interface FooterProps {}

const Footer: React.FC = () => {
  return (
    <Box component="footer" sx={{ backgroundColor: 'black', color: 'white', py: 3 }}>
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Research column */}
          <Grid item xs={12} sm={3}>
            <Typography variant="h6">Research</Typography>
            <Box component="nav" aria-label="Research links">
              <Link href="#" color="inherit" underline="hover">Overview</Link><br />
              <Link href="#" color="inherit" underline="hover">Index</Link><br />
              <Link href="#" color="inherit" underline="hover">GPT-4</Link><br />
              <Link href="#" color="inherit" underline="hover">DALL·E 3</Link>
            </Box>
          </Grid>

          {/* API column */}
          <Grid item xs={12} sm={3}>
            <Typography variant="h6">API</Typography>
            <Box component="nav" aria-label="API links">
              <Link href="#" color="inherit" underline="hover">Overview</Link><br />
              <Link href="#" color="inherit" underline="hover">Data privacy</Link><br />
              <Link href="#" color="inherit" underline="hover">Pricing</Link><br />
              <Link href="#" color="inherit" underline="hover">Docs</Link>
            </Box>
          </Grid>

          {/* ChatGPT column */}
          <Grid item xs={12} sm={3}>
            <Typography variant="h6">ChatGPT</Typography>
            <Box component="nav" aria-label="ChatGPT links">
              <Link href="#" color="inherit" underline="hover">Overview</Link><br />
              <Link href="#" color="inherit" underline="hover">Enterprise</Link><br />
              <Link href="#" color="inherit" underline="hover">Try ChatGPT</Link>
            </Box>
          </Grid>

          {/* Company column */}
          <Grid item xs={12} sm={3}>
            <Typography variant="h6">Company</Typography>
            <Box component="nav" aria-label="Company links">
              <Link href="#" color="inherit" underline="hover">About</Link><br />
              <Link href="#" color="inherit" underline="hover">Blog</Link><br />
              <Link href="#" color="inherit" underline="hover">Careers</Link><br />
              <Link href="#" color="inherit" underline="hover">Charter</Link><br />
              <Link href="#" color="inherit" underline="hover">Security</Link><br />
              <Link href="#" color="inherit" underline="hover">Customer stories</Link><br />
              <Link href="#" color="inherit" underline="hover">Safety</Link>
            </Box>
          </Grid>
        </Grid>

        <Box mt={4} sx={{ borderTop: '1px solid grey', pt: 2 }}>
          <Grid container justifyContent="space-between" alignItems="center">
            <Grid item>
              <Typography variant="body2">
                Plug and Play XAI © 2021 
              </Typography>
            </Grid>
            <Grid item>
              <Box component="nav" aria-label="Footer links">
                <Link href="#" color="inherit" underline="hover">Terms & policies</Link> | 
                <Link href="#" color="inherit" underline="hover">Privacy policy</Link> | 
                <Link href="#" color="inherit" underline="hover">Brand guidelines</Link> | 
                <Link href="#" color="inherit" underline="hover">Back to top ↑</Link>
              </Box>
            </Grid>
            <Grid item>
              <Box component="nav" aria-label="Social links">
                <Link href="#" color="inherit" underline="hover">GitHub</Link> | 
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;

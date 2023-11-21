import React, { useState, useEffect } from 'react';
import { Container, Card, CardContent, Typography, Grid } from '@mui/material';

const AboutPage = () => {
  const [aboutData, setAboutData] = useState(null);

  useEffect(() => {
    // Fetch the JSON file
    fetch('/src/assets/mockup/manifest.json') // Adjust the path as needed
      .then(response => response.json())
      .then(data => setAboutData(data));
  }, []);

  if (!aboutData) {
    return <div>Loading...</div>;
  }

  return (
    <Container sx={{ margin : 2 }}>
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h4" gutterBottom>
                {aboutData.packageManifest}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {Object.entries(aboutData.coreComponents).map(([key, value]) => (
          <Grid item xs={12} sm={6} md={4} key={key}>
            <Card>
              <CardContent>
                <Typography variant="h6">{key}</Typography>
                <Typography variant="body1">{value}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5">Vision</Typography>
              <Typography variant="body1">{aboutData.visionAndStrengths.vision}</Typography>

              {Object.entries(aboutData.visionAndStrengths.strengths).map(([key, value]) => (
                <Typography variant="body2" key={key}>
                  {value}
                </Typography>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default AboutPage;

import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  Typography, Card, CardContent, CardHeader, Box, Alert,
  FormControlLabel, Toolbar, Collapse, Button } from '@mui/material';
import { RootState } from '../app/store'; // Import your RootState type
import NetworkGraph from './ModelGraph';
import TreeGraph from './ModelGraph';

const ModelInfoComponent: React.FC = () => {
  const projectId = "test_project"; // Replace with your actual project ID
  const projectData = useSelector((state: RootState) => {
    return state.projects.data.find(project => project.id === projectId);
  });
  const [expanded, setExpanded] = useState<{ [key: string]: boolean }>({});
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [links, setLinks] = React.useState<any[]>([]);

  useEffect(() => {
    if (projectData?.experiments[0].model.nodes && projectData?.experiments[0].model.edges) {
      const deepCopy = JSON.parse(JSON.stringify(projectData?.experiments[0].model));
      setNodes(deepCopy.nodes);
      setLinks(deepCopy.edges);
    }
  }
  , [projectData]);



  const handleCollapse = (index: string | number) => {
    setExpanded(prevExpanded => ({
      ...prevExpanded,
      [index]: !prevExpanded[index]
    }));
  };
  

  const isNoModelDetected = projectData?.experiments.every((experiment) => {
    return !experiment.modelDetected;
    // return true; // For testing

  });

  return (
    <Box sx={{ m: 1 }}>
      <Typography variant='h1'> Model Detection Result </Typography>
      <Box sx={{ mt: 4 }}>
        {isNoModelDetected ? (
          <Box sx={{ m: 5, minHeight: "50px" }}>
            <Card>
              <CardContent>
                <Alert severity="warning">No available experiment. Try Again.</Alert>
              </CardContent>
            </Card>
          </Box>
        ) : (
          projectData?.experiments.map((experiment, index) => {
            const toolbarStyle = {
              backgroundColor: experiment.modelDetected ? 'green' : 'red',
              color: 'white',
            };
            return (
              <Box key={index} sx={{ m: 1 }}>
                <Card>
                  <CardHeader title={experiment.name} />
                  <Toolbar style={toolbarStyle}>
                    <Typography variant='h6'>
                      {experiment.modelDetected ? 'Model Detected' : 'Model Not Detected'}
                    </Typography>
                  </Toolbar>
                  <CardContent>
                    {experiment.modelDetected ? (
                      <Box sx={{ m: 1 }}>
                        <Typography variant='body1'> Model: {experiment.model.name} </Typography>
                        <Typography variant='body1'> Availalbe XAI Algorithms: </Typography>
                        {experiment.explainers.map((explainer, index) => (
                          <Typography key={index}> {explainer.name} </Typography>
                        ))}
                        <Button onClick={() => handleCollapse(index)}> View Model </Button>
                        <Collapse in={expanded[index]} timeout="auto" unmountOnExit>
                          <Typography variant='body1'> Model Structure </Typography>
                          <TreeGraph nodes={nodes} links={links} />
                          
                        </Collapse>
                      </Box>
                    ) : (
                      <Box sx={{ m: 5, minHeight: "50px" }}>
                        <Card>
                          <CardContent>
                            <Alert severity="warning"> We cannot identify Your Model Structure. Only torch.nn.Module Based model can be detected.</Alert>
                          </CardContent>
                        </Card>
                      </Box>
                    )
                  }
                    
                  </CardContent>
                </Card>
              </Box>
            );
          })
        )}
      </Box>
    </Box>
  );
};

export default ModelInfoComponent;
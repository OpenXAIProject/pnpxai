import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  Typography, Card, CardContent, CardHeader, Box, Alert,
  Toolbar, Collapse, Button,
  Tabs, Tab, Grid
} from '@mui/material';
import { RootState } from '../app/store'; // Import your RootState type
import TreeGraph from './ModelGraph';

const ModelInfoComponent: React.FC = () => {
  const projectId = "test_project"; // Replace with your actual project ID
  const projectData = useSelector((state: RootState) => {
    return state.projects.data.find(project => project.id === projectId);
  });
  const [expanded, setExpanded] = useState<{ [key: string]: boolean }>({});
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [links, setLinks] = React.useState<any[]>([]);
  const [value, setValue] = React.useState(0); // It should be extended to save data for each experiment

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
  });

  interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
  }

  function CustomTabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
  
    return (
      <div
        role="tabpanel"
        hidden={value !== index}
        id={`simple-tabpanel-${index}`}
        aria-labelledby={`simple-tab-${index}`}
        {...other}
      >
        {value === index && (
          <Box sx={{ p: 3 }}>
            <Typography>{children}</Typography>
          </Box>
        )}
      </div>
    );
  }
  
  function a11yProps(index: number) {
    return {
      id: `simple-tab-${index}`,
      'aria-controls': `simple-tabpanel-${index}`,
    };
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  }

  
  return (
    <Box sx={{ m: 1 }}>
      <Typography variant='h1'> Model Architecture Detection Results </Typography>
      <Box sx={{ mt: 3, mb: 3, ml: 1, pb: 3, borderBottom: 1, minHeight: "600px" }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Box sx={{ borderRight: 1, borderColor: 'divider', m: 2 }}>
            <Typography variant='h3'> SIDEBAR AREA </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={9}>
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
                color: experiment.modelDetected ? 'green' : 'black',
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

                          <Box>
                            <Typography variant='h3'> Model architecture was detected successfully </Typography>
                          </Box>

                          <Button onClick={() => handleCollapse(index)}> View Model </Button>
                            <Collapse in={expanded[index]} timeout="auto" unmountOnExit>
                              <Box sx={{ m: 2}}>
                                <Tabs value={value} onChange={handleTabChange}>
                                  <Tab label="Text View" />
                                  {/* <Tab label="Tree View" /> */}
                                </Tabs>
                              </Box>
                              <CustomTabPanel value={value} index={0}>
                                {nodes.map((node: any, index: number) => (
                                  <Typography key={index}> {node.id} </Typography>
                                ))}
                              </CustomTabPanel>
                              {/* <CustomTabPanel value={value} index={1}>
                                Tree View
                                <TreeGraph nodes={nodes} links={links} />
                              </CustomTabPanel> */}
                            </Collapse>

                        </Box>
                      ) : (
                        <Box sx={{ m: 5, minHeight: "50px" }}>
                          <Card>
                            <CardContent>
                              <Alert severity="warning"> Cannot detect model architecture. Try other models. </Alert>
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
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default ModelInfoComponent;
import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  Typography, Card, CardContent, CardHeader, Box, Alert,
  Toolbar, Collapse, Button,
  Tabs, Tab, Grid
} from '@mui/material';
import { RootState } from '../app/store'; // Import your RootState type
import TreeGraph from './ModelGraph';
import { Experiment } from '../app/types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const ModelInfoComponent: React.FC<{ experiment: Experiment, showModel: boolean}> = ({ experiment, showModel }) => {
  // const [expanded, setExpanded] = useState<boolean>(showModel ? true : false);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [links, setLinks] = React.useState<any[]>([]);
  const [value, setValue] = React.useState(0); // It should be extended to save data for each experiment
  const [buttonName, setButtonName] = React.useState<string>(showModel ? 'Hide Model Architecture':'Show Model Architecture');
  const toolbarStyle = {
    color: experiment.id ? 'green' : 'black',
  };

  useEffect(() => {
    if (experiment.model.nodes && experiment.model.edges) {
      const deepCopy = JSON.parse(JSON.stringify(experiment.model));
      setNodes(deepCopy.nodes);
      setLinks(deepCopy.edges);
    }
  }
  , [experiment]);

  const handleCollapse = () => {
    setExpanded(!expanded);
    setButtonName(buttonName === 'Show Model Architecture' ? 'Hide Model Architecture' : 'Show Model Architecture');
  };
  

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
            {children}
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
      <Box sx={{ mt: 3, mb: 3, ml: 1, pb: 3, borderBottom: 1, minHeight: "600px" }}>
      <Card>
        <Grid container spacing={2}>
          {/* Side Bar Area */}
          <Grid item xs={12} md={3} sx={{borderRight: 1, borderColor: 'divider'}}>
            <Box sx={{ m: 2 }}>
            <Box sx={{ mb: 3 }}>
              <Typography variant='h6'> Experiment Name </Typography>
              <Typography variant='h3'> {experiment.name} </Typography>
            </Box>
            <Box sx={{ mb: 3 }}>
              <Typography variant='h6'> Model Name </Typography>
              <Typography variant='h3'> {experiment.model.name} </Typography>
            </Box>
            <Box sx={{ mb: 3 }}>
              <Typography variant='h6'> Model Detection Result </Typography>
              <Typography variant='body1' style={toolbarStyle}> Model Detected </Typography>
            </Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={9}>
            <Box  sx={{ m: 1 }}>
              <Box sx={{ m : 3}}>
                <Typography variant='h2'> Model Information </Typography>
              </Box>
              <Box sx={{ m: 1 }}>
                <Box sx={{ m: 2 }}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant='h3'> Recommended Explainers </Typography>
                  </Box>
                  {experiment.explainers.map((explainer, index) => (
                    <Box key={index} sx={{ m: 1 }}>
                      <Card sx={{ p: 2 }}>
                        <Typography> {explainer.name} </Typography>
                      </Card>
                    </Box>
                  ))}
                </Box>
                <Box sx={{ m: 2 }}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant='h3'> Model Architecture </Typography>
                  </Box>
                  <Button onClick={() => handleCollapse()}> {buttonName} </Button>
                    <Collapse in={expanded} timeout="auto" unmountOnExit>
                      <Box sx={{ m: 2}}>
                        <Tabs value={value} onChange={handleTabChange}>
                          <Tab label="Text View" />
                          {/* <Tab label="Tree View" /> */}
                        </Tabs>
                      </Box>
                      <CustomTabPanel value={value} index={0}>
                        {nodes.map((node: any, index: number) => (
                          <Typography key={index} component='div'> {node.operator} </Typography> 
                        ))}
                      </CustomTabPanel>
                      {/* <CustomTabPanel value={value} index={1}>
                        Tree View
                        <TreeGraph nodes={nodes} links={links} />
                      </CustomTabPanel> */}
                    </Collapse>
                </Box>
              </Box>
            </Box>
          </Grid>
        </Grid>
        </Card>
      </Box>
    </Box>
  );
};

export default ModelInfoComponent;
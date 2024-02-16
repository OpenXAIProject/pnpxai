import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../app/store';
import { setCurrentProject, setColorMap } from '../../features/projectSlice';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem, Popover, IconButton, Tooltip, Typography, List, ListItem } from '@mui/material';
import logo from '../../assets/images/SVG/XAI-Top-PnP.svg';
import { useLocation } from 'react-router-dom';
import HelpRoundedIcon from '@mui/icons-material/HelpRounded';
import SettingsIcon from '@mui/icons-material/Settings';
import ColorScales from '../../assets/styles/colorScale.json';

interface HelpText {
  [key: string]: string;
}

const NavBar: React.FC = () => {
  const helptext: HelpText = {
    "Correctness" : "the truthfulness/reliability of explanations about a prediction model (AI model). That is, it indicates how truthful the explanation is compared to the operation of the black box model.",
    "Continuity" : "how continuous (i.e., smooth) an explanation is. An explanation function with high continuity ensures that small changes in the input do not bring about significant changes in the explanation.",
    "Compactness" : "the size/amount of an explanation. It ensures that complex and redundant explanations that are difficult to understand are not presented.",
    // "Completeness" : " the extent to which a prediction model (AI model) is explained. Providing 'the whole truth' of the black box model represents high completeness, but a good explanation should balance conciseness and correctness.",
  }

  const colorMaps = Object.keys(ColorScales);

  const routes = [
    {
      path: "/model-info",
      name: "Experiment Information"
    },
    {
      path: "/model-explanation",
      name: "Local Explanation"
    }
  ]

  
  const projectsData = useSelector((state: RootState) => state.projects.data);
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const projects = projectsData?.map(project => project.id) || [];
  const [projectAnchorEl, setProjectAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedMenu, setSelectedMenu] = useState<number | null>(null);
  const location = useLocation();

  const dispatch = useDispatch();

  const handleSelectMenu = (menuKey: number) => {
    setSelectedMenu(menuKey);
  };

  const handleProjectMenuButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setProjectAnchorEl(event.currentTarget);
  };

  const handleProjectMenuClose = () => {
    setProjectAnchorEl(null);
  };

  const handleSelectProject = (projectId: string) => {
    setSelectedProject(projectId);
    dispatch(setCurrentProject(projectId));
    handleProjectMenuClose();
  };

  useEffect(() => {
    const currentPath = window.location.hash.split('#')[1];
    const currentMenu = routes.findIndex(route => route.path === currentPath);
    if (currentMenu !== -1) {
      setSelectedMenu(currentMenu);
    } else {
      setSelectedMenu(0);
    }
  }
  , [location]);

  useEffect(() => {
    setSelectedProject(projectId);
  }
  , [projectId]);


  const [helperAnchor, setHelperAnchor] = useState(null);
  const [settingAnchor, setSettingAnchor] = useState(null);
  const [selectedColorMap, setSelectedColorMap] = useState(colorMaps[0]); // Default colormap


  const handleHelperClick = (event: any) => {
    setHelperAnchor(event.currentTarget);
  };

  const handleHelperClose = () => {
    setHelperAnchor(null);
  };

  const handleSettingClick = (event: any) => {
    setSettingAnchor(event.currentTarget);
  }

  const handleSettingClose = () => {
    setSettingAnchor(null);
  }

  const handleColorMapChange = (colorMap: any) => {
    setSelectedColorMap(colorMap);
    dispatch(setColorMap(colorMap));
    handleSettingClose(); // Close the popover after selection
  };
  const helperOpen = Boolean(helperAnchor);
  const settingOpen = Boolean(settingAnchor);
  const helperId = helperOpen ? 'helper-popover' : undefined;
  const settingId = settingOpen ? 'setting-popover' : undefined;


  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 30, height: '40px' }} />
          </Link>
          <Box sx={{pr : 2, borderRight : 1}}>
            <Button style={{ color: 'inherit' }} onClick={handleProjectMenuButtonClick}>
              Projects : {selectedProject}
            </Button>
            <Menu
              anchorEl={projectAnchorEl}
              open={Boolean(projectAnchorEl)}
              onClose={handleProjectMenuClose}
            >
              {projects.map((project, index) => (
                <MenuItem 
                  key={index} 
                  onClick={() => handleSelectProject(project)}
                  style={{ fontWeight: project === selectedProject ? 'bold' : 'normal' }}
                >
                  {project}
                </MenuItem>
              ))}
            </Menu>
          </Box>
          
          {/* Left of the menu bar */}
          {routes.map((route, index) => {
            return (
              <Button 
                key={index}
                component={Link} 
                to={route.path} 
                onClick={() => handleSelectMenu(index)} 
                style={{ 
                  marginLeft: 20, 
                  color: 'inherit', 
                  fontWeight: selectedMenu === index ? 'bold' : 'normal'
                }}
              >
                {route.name}
              </Button>
            )
          })}
          
          <Box sx={{ flexGrow: 1 }} />

          {/* Right of the menu bar */}
          <Box sx={{ mr : 1}}>
            <Tooltip title="Settings">
              <IconButton aria-describedby={settingId} onClick={handleSettingClick}>
                <SettingsIcon sx={{ color : 'white'}}/>
              </IconButton>
            </Tooltip>
            <Popover
              id={settingId}
              open={settingOpen}
              anchorEl={settingAnchor}
              onClose={handleSettingClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              PaperProps={{
                sx: {
                  maxWidth: 600, // Set the maximum width of the Popover
                }
              }}
            >
              <Typography sx={{ p: 2 }}> Color Maps </Typography>
              {/* ColorMap buttons */}
              <Box sx={{ p: 2 }}>
                {colorMaps.map((colorMap) => (
                  <Button
                    key={colorMap}
                    variant={selectedColorMap === colorMap ? 'contained' : 'outlined'}
                    onClick={() => handleColorMapChange(colorMap)}
                    sx={{ margin: 0.5 }}
                  >
                    {colorMap}
                  </Button>
                ))}
              </Box>
            </Popover>
          </Box>

          <Box sx={{ mr : 1 }}>
            <Tooltip title="Meaning of evaluation metric">
              <IconButton aria-describedby={helperId} onClick={handleHelperClick}>
                <HelpRoundedIcon sx={{ color : 'white'}}/>
              </IconButton>
            </Tooltip>
            <Popover
              id={helperId}
              open={helperOpen}
              anchorEl={helperAnchor}
              onClose={handleHelperClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              PaperProps={{
                sx: {
                  maxWidth: 600, // Set the maximum width of the Popover
                  // Add any additional styles here
                }
              }}
            >
              <Typography sx={{ p: 2 }}> Meaning of Evaluation Metric </Typography>
              <List sx={{ p: 1 }}>
                {Object.keys(helptext).map((key, index) => (
                    <ListItem key={index}>
                    <Typography key={index} sx={{ p: 1 }}>
                    <strong>{key}</strong> evaluates {helptext[key]}
                    </Typography>
                    </ListItem>
                  ))}
              </List>
            </Popover>
          </Box>
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../app/store';
import { setCurrentProject, setColorMap } from '../features/globalState';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem, Popover, IconButton, Tooltip, Typography, List, ListItem } from '@mui/material';
import logo from '../../assets/images/SVG/XAI-Top-PnP.svg';
import { useLocation } from 'react-router-dom';
import HelpRoundedIcon from '@mui/icons-material/HelpRounded';
import SettingsIcon from '@mui/icons-material/Settings';
import ColorScalesData from '../assets/styles/colorScale.json';
import { ColorScales } from '../app/types';
import { helptext, routes } from '../components/util';

const NavBar: React.FC = () => {
  const projects = useSelector((state: RootState) => state.global.projects);
  const projectId = useSelector((state: RootState) => state.global.status.currentProject);
  const expId = useSelector((state: RootState) => state.global.status.currentExp);
  const cache = useSelector((state: RootState) => state.global.cache.filter((item) => item.projectId === projectId && item.expId === expId)[0]);
  const colorScales: ColorScales = ColorScalesData;
  const colorMap = cache?.config.colorMap;



  const [projectAnchorEl, setProjectAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedMenu, setSelectedMenu] = useState<number | null>(null);
  const [helperAnchor, setHelperAnchor] = useState(null);
  const [settingAnchor, setSettingAnchor] = useState(null);

  const helperOpen = Boolean(helperAnchor);
  const settingOpen = Boolean(settingAnchor);
  const helperId = helperOpen ? 'helper-popover' : undefined;
  const settingId = settingOpen ? 'setting-popover' : undefined;



  const location = useLocation();
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
    dispatch(setCurrentProject(projectId));
    handleProjectMenuClose();
  };

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
    dispatch(setColorMap({ projectId: projectId, expId: expId, colorMap: colorMap }));
    handleSettingClose(); // Close the popover after selection
  };
  


  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 30, height: '40px' }} />
          </Link>
          <Box sx={{pr : 2, borderRight : 1}}>
            <Button style={{ color: 'inherit' }} onClick={handleProjectMenuButtonClick}>
              Projects : {projectId}
            </Button>
            <Menu
              anchorEl={projectAnchorEl}
              open={Boolean(projectAnchorEl)}
              onClose={handleProjectMenuClose}
            >
              {projects.map((project, index) => (
                <MenuItem 
                  key={index} 
                  onClick={() => handleSelectProject(project.id)}
                  style={{ fontWeight: project.id === projectId ? 'bold' : 'normal' }}
                >
                  {project.id}
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
              <Typography sx={{ p: 2 }}> Sequential Color Maps </Typography>
              {/* ColorMap buttons */}
              <Box sx={{ p: 2 }}>
                {Object.keys(colorScales.seq).map((cmap: any) => (
                  <Button
                    key={cmap}
                    variant={colorMap.seq === cmap ? 'contained' : 'outlined'}
                    onClick={() => handleColorMapChange({'seq' : cmap, 'diverge' : colorMap.diverge})}
                    sx={{ margin: 0.5 }}
                  >
                    {cmap}
                  </Button>
                ))}
              </Box>
              <Typography sx={{ p: 2 }}> Diverge Color Maps </Typography>
              {/* ColorMap buttons */}
              <Box sx={{ p: 2 }}>
                {Object.keys(colorScales.diverge).map((cmap:any) => (
                  <Button
                    key={cmap}
                    variant={colorMap.diverge === cmap ? 'contained' : 'outlined'}
                    onClick={() => handleColorMapChange({'seq' : colorMap.seq, 'diverge' : cmap})}
                    sx={{ margin: 0.5 }}
                  >
                    {cmap}
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

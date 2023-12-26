import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../app/store';
import { setCurrentProject } from '../../features/projectSlice';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem, Popover, IconButton, Tooltip, Typography } from '@mui/material';
import logo from '../../assets/images/logo.svg';
import { useLocation } from 'react-router-dom';
import HelpRoundedIcon from '@mui/icons-material/HelpRounded';


const NavBar: React.FC = () => {
  const helptext = "Correctness evaluates the truthfulness/reliability of explanations about a prediction model (AI model). That is, it indicates how truthful the explanation is compared to the operation of the black box model. Completeness assesses the extent to which a prediction model (AI model) is explained. Providing 'the whole truth' of the black box model represents high completeness, but a good explanation should balance conciseness and correctness. Continuity evaluates how continuous (i.e., smooth) an explanation is. An explanation function with high continuity ensures that small changes in the input do not bring about significant changes in the explanation. Compactness assesses the size/amount of an explanation. It ensures that complex and redundant explanations that are difficult to understand are not presented."
  const tasks = [
    "Image Classification",
    "Tabular Data Classification",
    "Time Series Analysis",
    "Text Classification",
  ];

  
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
  const [taskAnchorEl, setTaskAnchorEl] = useState<null | HTMLElement>(null);
  const [task, setTask] = useState<string>("");
  const [projectAnchorEl, setProjectAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedMenu, setSelectedMenu] = useState<number | null>(null);
  const location = useLocation();

  const dispatch = useDispatch();

  const handleSelectMenu = (menuKey: number) => {
    setSelectedMenu(menuKey);
  };

  const handleTaskMenuButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setTaskAnchorEl(event.currentTarget);
  }

  const handleTaskMenuClose = () => {
    setTaskAnchorEl(null);
  }

  const handleTaskChange = (event: any) => {
    // setTask(event.target.value);
    handleTaskMenuClose();
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
    const currentPath = window.location.pathname;
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
    if (projectId) {
      setTask(tasks[0]);
    }
  }
  , [projectId]);


  const [anchorEl, setAnchorEl] = useState(null);

  const handleClick = (event: any) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);
  const id = open ? 'simple-popover' : undefined;


  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          <Box sx={{borderRight : 1}}>
            <Button style={{ color: 'inherit' }} onClick={handleTaskMenuButtonClick}>
              Task : {task}
            </Button>
            <Menu
              anchorEl={taskAnchorEl}
              open={Boolean(taskAnchorEl)}
              onClose={handleTaskMenuClose}
            >
              {tasks.map((task, index) => (
                <MenuItem 
                  key={index}
                  onClick={() => handleTaskChange(task)}
                  style={{ fontWeight: task === tasks[0] ? 'bold' : 'normal' }}
                >
                  {task}
                </MenuItem>
              ))}
            </Menu>
          </Box>

          <Box sx={{borderRight : 1}}>
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
          <Tooltip title="Meaning of evaluation metric">
            <IconButton aria-describedby={id} onClick={handleClick}>
              <HelpRoundedIcon />
            </IconButton>
          </Tooltip>
          <Popover
            id={id}
            open={open}
            anchorEl={anchorEl}
            onClose={handleClose}
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
            <Typography sx={{ p: 2 }}> {helptext} </Typography> 
          </Popover>
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

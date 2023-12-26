import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../app/store';
import { setCurrentProject } from '../../features/projectSlice';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem } from '@mui/material';
import logo from '../../assets/images/logo.svg';
import { useLocation } from 'react-router-dom';


const NavBar: React.FC = () => {
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
          
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

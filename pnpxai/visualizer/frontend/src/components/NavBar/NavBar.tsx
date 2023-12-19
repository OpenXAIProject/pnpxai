import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../app/store';
import { setCurrentProject } from '../../features/projectSlice';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem } from '@mui/material';
import logo from '../../assets/images/logo.svg';

const NavBar: React.FC = () => {
  const routes = [
    {
      path: "/model-info",
      name: "Model Architecture Information"
    },
    {
      path: "/model-explanation",
      name: "Local Explanation"
    }
  ]

  const projectsData = useSelector((state: RootState) => state.projects.data);
  const projectId = useSelector((state: RootState) => state.projects.currentProject.id);
  const projects = projectsData?.map(project => project.id) || [];
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedProject, setSelectedProject] = useState<string>("");
  const [selectedMenu, setSelectedMenu] = useState<number | null>(null);

  const dispatch = useDispatch();

  const handleSelectMenu = (menuKey: number) => {
    setSelectedMenu(menuKey);
  };

  const handleMenuButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setMenuAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  const handleSelectProject = (projectId: string) => {
    setSelectedProject(projectId);
    dispatch(setCurrentProject(projectId));
    handleMenuClose();
  };

  useEffect(() => {
    setSelectedMenu(routes.findIndex(route => route.path === window.location.pathname));
  }
  , []);

  useEffect(() => {
    setSelectedProject(projectId);
  }
  , [projectId]);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          {/* Left of the menu bar */}
          {routes.map((route, index) => {
            return (
              <Button 
                key={index}
                component={Link} 
                to={route.path} 
                onClick={() => handleSelectMenu(index)} 
                style={{ 
                  marginRight: 50, 
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
          <Button style={{ color: 'inherit' }} onClick={handleMenuButtonClick}>
            Projects : {selectedProject}
          </Button>
          <Menu
            anchorEl={menuAnchorEl}
            open={Boolean(menuAnchorEl)}
            onClose={handleMenuClose}
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
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

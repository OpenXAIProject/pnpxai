import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, Button, Menu, MenuItem } from '@mui/material';
import logo from '../../assets/images/logo.svg';

const NavBar: React.FC = () => {
  const projects = ['test project', 'project2', 'project3'];
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedProject, setSelectedProject] = useState<string>(projects[0]);
  const [selectedMenu, setSelectedMenu] = useState<number | null>(0);

  const handleSelectMenu = (menuKey: number) => {
    setSelectedMenu(menuKey);
  };

  const handleMenuButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setMenuAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  const handleSelectProject = (project: string) => {
    // setSelectedProject(project);
    handleMenuClose();
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          {/* Left of the menu bar */}
          <Button 
            component={Link} 
            to="/model-info" 
            onClick={() => handleSelectMenu(0)} 
            style={{ 
              marginRight: 50, 
              color: 'inherit', 
              fontWeight: selectedMenu === 0 ? 'bold' : 'normal'
            }}
          >
            Model Architecture Information
          </Button>
          <Button 
            component={Link} 
            to="/model-explanation" 
            onClick={() => handleSelectMenu(1)} 
            style={{ 
              marginRight: 50, 
              color: 'inherit', 
              fontWeight: selectedMenu === 1 ? 'bold' : 'normal' 
            }}
          >
            Local Explanation
          </Button>

          <Box sx={{ flexGrow: 1 }} />
          {/* Right of the menu bar */}
          <Button style={{ color: 'inherit' }} onClick={handleMenuButtonClick}>
            Projects
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

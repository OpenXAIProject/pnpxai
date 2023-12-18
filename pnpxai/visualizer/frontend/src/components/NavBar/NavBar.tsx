import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Box, AppBar, Toolbar, FormControl, InputLabel, Select, Menu, MenuItem, Button } from '@mui/material';
import logo from '../../assets/images/logo.svg';

const NavBar: React.FC = () => {
  const projects = ['test project', 'project2', 'project3'];
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);

  const handleMenuButtonClick = (event:any) => {
    setMenuAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  const handleSelectProject = (project:any) => {
    // setSelectedItem(project); // If you want to keep track of the selected item
    handleMenuClose();
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          <Button style={{ marginRight: 50, color: 'inherit' }} onClick={handleMenuButtonClick}>
            Projects
          </Button>
          <Menu
            anchorEl={menuAnchorEl}
            open={Boolean(menuAnchorEl)}
            onClose={handleMenuClose}
          >
            {projects.map((project, index) => (
              // <MenuItem key={index} onClick={() => handleSelectProject(project)} component={Link} to={`project/${project}`}>
              <MenuItem key={index} onClick={() => handleSelectProject(project)}>
                {project}
              </MenuItem>
            ))}
          </Menu>
          <Button component={Link} to="/model-info" style={{ marginRight: 50, color: 'inherit' }}>
            Model Architecture Information
          </Button>
          <Button component={Link} to="/model-explanation" style={{ marginRight: 50, color: 'inherit' }}>
            Local Explanation
          </Button>
          {/* Spacer to push the box to the right */}
          <Box sx={{ flexGrow: 1 }} />
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

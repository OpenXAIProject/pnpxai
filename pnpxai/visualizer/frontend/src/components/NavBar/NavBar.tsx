import React from 'react';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Toolbar from '@mui/material/Toolbar';
import logo from '../../assets/images/logo.svg';

const NavBar: React.FC = () => {

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
        <Toolbar>
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          <Button component={Link} to="/model-info" style={{ marginRight: 50, color: 'inherit' }}>
            Model Information
          </Button>
          <Button component={Link} to="/model-explanation" style={{ marginRight: 50, color: 'inherit' }}>
            Model Explanation
          </Button>
          {/* Spacer to push the box to the right */}
          <Box sx={{ flexGrow: 1 }} />
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

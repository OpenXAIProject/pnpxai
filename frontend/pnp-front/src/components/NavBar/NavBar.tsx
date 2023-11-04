// src/components/NavBar/NavBar.tsx
import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import { useDispatch } from 'react-redux';
import { toggle } from '../../features/sidebar/sidebarSlice'; // adjust the import path if needed

const NavBar: React.FC = () => {
  const dispatch = useDispatch();

  const handleToggleSidebar = () => {
    dispatch(toggle());
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            size="large"
            edge="start"
            color="inherit"
            aria-label="menu"
            sx={{ mr: 2 }}
            onClick={handleToggleSidebar}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Plug and Play XAI
          </Typography>
          {/* Add additional navigation items here */}
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

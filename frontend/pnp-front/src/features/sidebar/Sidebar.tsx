// src/components/SideBar/Sidebar.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'; // Import for close icon
import { RootState } from '../../app/store'; // Adjust the import according to your store location
import { toggle } from './sidebarSlice';

const Sidebar: React.FC = () => {
  const dispatch = useDispatch();
  const isOpen = useSelector((state: RootState) => state.sidebar.isOpen);

  const handleToggle = () => {
    dispatch(toggle());
  };

  return (
    <div>
      <Drawer
        variant="temporary"
        anchor="left"
        open={isOpen}
        onClose={handleToggle} // Allows the drawer to close when clicking outside of it
        sx={{
          '& .MuiDrawer-paper': { width: 240, boxSizing: 'border-box' },
        }}
        ModalProps={{ // This helps to keep the drawer open even when clicking inside it
          keepMounted: true,
        }}
      >
        <div>
          <IconButton onClick={handleToggle}>
            <ChevronLeftIcon />
          </IconButton>
        </div>
        <List>
          <ListItem button component={Link} to="/" onClick={handleToggle}>
            <ListItemIcon>
              <HomeIcon />
            </ListItemIcon>
            <ListItemText primary="Home" />
          </ListItem>
          <ListItem button component={Link} to="/about" onClick={handleToggle}>
            <ListItemIcon>
              <InfoIcon />
            </ListItemIcon>
            <ListItemText primary="About" />
          </ListItem>
          {/* Add more ListItems here for additional navigation links */}
        </List>
      </Drawer>
    </div>
  );
};

export default Sidebar;

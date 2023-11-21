import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import Popover from '@mui/material/Popover';
import logo from '../../assets/images/logo.svg'; // Adjust the path as necessary
import { Link, useNavigate } from 'react-router-dom'; // Import useNavigate

const NavBar: React.FC = () => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const navigate = useNavigate();

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleListItemClick = (path: string) => {
    navigate(path);
    handleClose();
  };

  const open = Boolean(anchorEl);
  const id = open ? 'model-description-popover' : undefined;

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ bgcolor: `primary.main` }}>
        <Toolbar>
          {/* Logo */}
          <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <img src={logo} alt="Logo" style={{ marginRight: 50, height: '50px' }} />
          </Link>
          
          {/* NavBar Items */}
          <Button component={Link} to="/model-info" style={{ marginRight : 50, color: 'inherit' }}>
            Model Information(모델 정보)
          </Button>
          <Button aria-describedby={id} onClick={handleClick} style={{ marginRight : 50, color: 'inherit' }}>
            Model Explanation(모델 설명)
          </Button>
          <Popover
              id={id}
              open={open}
              anchorEl={anchorEl}
              onClose={handleClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
            >
              <List component="nav">
                <ListItem button onClick={() => handleListItemClick('/model-explanation')}>
                  (Auto Explain) 자동으로 설명하기
                </ListItem>
                <ListItem button onClick={() => handleListItemClick('/model-explanation/manual')}>
                  (Manual Explain) 수동으로 설명하기
                </ListItem>
              </List>
            </Popover>
          <Button component={Link} to="/about" style={{ marginRight : 50, color: 'inherit' }}>
            Explainable XAI
          </Button>
        </Toolbar>
      </AppBar>
    </Box>
  );
};

export default NavBar;

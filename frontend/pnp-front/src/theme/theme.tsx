// src/theme/theme.ts
import { createTheme } from '@mui/material/styles';

// Create a theme instance.
const theme = createTheme({
  palette: {
    primary: {
      main: '#556cd6',
    },
    secondary: {
      main: '#19857b',
    },
    error: {
      main: '#ff0000',
    },
    background: {
      default: '#fff',
      paper: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: [
      'Open Sans', // your chosen font
      'Roboto', // default MUI font
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','), // Ensure fonts with spaces have " " surrounding it.
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    // You can continue customizing the typography for h3, h4, h5, h6, body1, body2, etc.
  },
  // You can add other customizations such as spacing, breakpoints, etc.
});

export default theme;

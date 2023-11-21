// src/theme/theme.ts
import { createTheme } from '@mui/material/styles';
import '/src/assets/fonts/fonts.css'

// Create a theme instance.
const theme = createTheme({
  palette: {
    primary: {
      main: '#283593',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#607D8B',
      contrastText: '#FFFFFF',
    },
    error: {
      main: '#D32F2F',
    },
    warning: {
      main: '#FFA000',
    },
    info: {
      main: '#1976D2',
    },
    success: {
      main: '#388E3C',
    },
    background: {
      default: '#ECEFF1',
      paper: '#CFD8DC',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
  },
  typography: {
    fontFamily: [
      '"Noto Sans KR"', // Korean font
      // '"Nanum Pen Script"', // Korean font
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
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
  },
});

export default theme;

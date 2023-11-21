// src/App.tsx
import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme/theme';
import AppRoutes from './routes/AppRoutes';
import CssBaseline from '@mui/material/CssBaseline';
import { RootState } from './app/store'; // make sure this path is correct
import { useSelector } from 'react-redux';
import Footer from './components/Footer/Footer';
import NavBar from './components/NavBar/NavBar';
import Layout from './layouts/layout';


const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
      <Layout>
        <AppRoutes />
      </Layout>
      </Router>
    </ThemeProvider>
  );
};

export default App;

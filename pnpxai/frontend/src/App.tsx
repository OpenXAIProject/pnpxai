// src/App.tsx
import React, { useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme/theme';
import AppRoutes from './routes/AppRoutes';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './layouts/layout';
import { useDispatch, useSelector } from 'react-redux';
import { fetchProjects } from './features/projectSlice';
import { RootState } from './app/store'; // Import the RootState type




const App: React.FC = () => {
  const dispatch = useDispatch()
  const isLoaded = useSelector((state: RootState) => state.projects.loaded);

  useEffect(() => {
    if (!isLoaded) {
      dispatch(fetchProjects() as any);
    }
  }, [dispatch, isLoaded]);


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

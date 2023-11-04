// src/App.tsx
import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme/theme';
import Sidebar from './features/sidebar/Sidebar';
import AppRouter from './routes/AppRouter';
import CssBaseline from '@mui/material/CssBaseline';
import { RootState } from './app/store'; // make sure this path is correct
import { useSelector } from 'react-redux';
import Footer from './components/Footer/Footer';
import NavBar from './components/NavBar/NavBar';


const App: React.FC = () => {
  const sidebarIsOpen = useSelector((state: RootState) => state.sidebar.isOpen);


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <NavBar />
        <Sidebar />
        <div style={{
          transform: sidebarIsOpen ? 'translateX(250px)' : 'translateX(0px)',
          transition: 'transform 0.3s ease-in-out'
        }}>
        <AppRouter />
        <Footer />
        </div>
      </Router>
    </ThemeProvider>

  );
};

export default App;

// src/layouts/Layout.tsx
import React from 'react';
import NavBar from '../components/NavBar/NavBar';
import Footer from '../components/Footer/Footer';
import { Container, Box } from '@mui/material';

type LayoutProps = {
  children: React.ReactNode;
};

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <>
      <NavBar />
      <Box>
        {children}
      </Box>
      <Footer />
    </>
  );
};

export default Layout;

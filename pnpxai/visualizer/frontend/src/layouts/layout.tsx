// src/layouts/Layout.tsx
import React from 'react';
import NavBar from './NavBar';
import Footer from './Footer';
import { Box } from '@mui/material';

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

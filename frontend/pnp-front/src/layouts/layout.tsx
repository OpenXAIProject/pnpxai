// src/layouts/Layout.tsx
import React from 'react';
import NavBar from '../components/NavBar/NavBar';
import Footer from '../components/Footer/Footer';
import { Box, Container } from '@mui/material';

type LayoutProps = {
  children: React.ReactNode;
};

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <>
      <NavBar />
      <Container maxWidth="lg">
        <main>{children}</main>
      </Container>
      <Footer />
    </>
  );
};

export default Layout;

// src/routes/AppRoutes.tsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HomePage from '../pages/HomePage';
import NotFoundPage from '../pages/NotFoundPage';
import ModelInfoPage from '../pages/ModelInfoPage';
import ManualExplainPage from '../pages/ManualExplainPage';
import AutoExplainPage from '../pages/AutoExplainPage';
import AboutPage from '../pages/AboutPage';

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/model-info" element={<ModelInfoPage/>} />
      <Route path="/model-explanation" element={<AutoExplainPage/>} />
      <Route path="/model-explanation/manual" element={<ManualExplainPage/>} />
      <Route path="/about" element={<AboutPage/>} />
      <Route path="/*" element={<NotFoundPage />} />
    </Routes>
  );
};

export default AppRoutes;


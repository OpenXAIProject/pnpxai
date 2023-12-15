// src/routes/AppRoutes.tsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import NotFoundPage from '../pages/NotFoundPage';
import ModelInfoPage from '../pages/ModelInfoPage';
import ExperimentPage from '../pages/ExperimentPage';
import TestPage from '../pages/testPage';

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<ModelInfoPage/>} />
      <Route path="/model-info" element={<ModelInfoPage/>} />
      <Route path="/model-explanation" element={<ExperimentPage/>} />
      <Route path="/test" element={<TestPage/>} />
      <Route path="/*" element={<NotFoundPage />} />
    </Routes>
  );
};

export default AppRoutes;


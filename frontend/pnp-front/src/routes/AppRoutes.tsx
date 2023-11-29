// src/routes/AppRoutes.tsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import NotFoundPage from '../pages/NotFoundPage';
import ModelInfoPage from '../pages/ModelInfoPage';
import AutoExplainPage from '../pages/AutoExplainPage';
import ExperimentPage from '../pages/ExperimentPage';

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/model-info" element={<ModelInfoPage/>} />
      <Route path="/model-explanation" element={<ExperimentPage/>} />
      <Route path="/*" element={<NotFoundPage />} />
    </Routes>
  );
};

export default AppRoutes;


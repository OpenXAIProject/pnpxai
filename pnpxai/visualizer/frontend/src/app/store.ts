// src/app/store.ts
import { configureStore } from '@reduxjs/toolkit';
import globalReducer from '../features/globalState';


export const store = configureStore({
  reducer: {
    global: globalReducer,
  },
});

export type AppDispatch = typeof store.dispatch
export type RootState = ReturnType<typeof store.getState>


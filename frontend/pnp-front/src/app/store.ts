// In your store configuration file
import { configureStore } from '@reduxjs/toolkit';
import experimentReducer from '../features/experimentSlice';

export const store = configureStore({
  reducer: {
    experiments: experimentReducer,
  },
});

export type AppDispatch = typeof store.dispatch
export type RootState = ReturnType<typeof store.getState>

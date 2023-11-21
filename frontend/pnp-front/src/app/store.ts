// In your store configuration file
import { configureStore } from '@reduxjs/toolkit';
import yourDataReducer from '../features/yourDataSlice';
import algorithmReducer from '../features/algorithmSlice';
import explainReducer from '../features/explainSlice';


export const store = configureStore({
  reducer: {
    data: yourDataReducer,
    algorithm: algorithmReducer,
    explain: explainReducer,
  },
});

export type AppDispatch = typeof store.dispatch
export type RootState = ReturnType<typeof store.getState>

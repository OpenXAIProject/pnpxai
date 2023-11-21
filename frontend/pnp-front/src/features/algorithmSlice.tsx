// src/features/algorithmSlice.tsx
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../app/store'; // Adjust the import path to your store file

export interface AlgorithmType {
    id: number;
    name: string;
    selected: boolean;
}

interface AlgorithmState {
    algorithms: AlgorithmType[];
}

const exampleAlgorithms: AlgorithmType[] = [
    { id: 1, name: 'Algorithm A', selected: false },
    { id: 2, name: 'Algorithm B', selected: false },
    { id: 3, name: 'Algorithm C', selected: false },
    // Add more algorithms as needed
];

const initialState: AlgorithmState = {
    algorithms: exampleAlgorithms,
};


export const algorithmSlice = createSlice({
    name: 'algorithm',
    initialState,
    reducers: {
        setAlgorithms: (state, action: PayloadAction<AlgorithmType[]>) => {
            state.algorithms = action.payload;
        },
        toggleAlgorithm: (state, action: PayloadAction<number>) => {
            const algorithm = state.algorithms.find(algo => algo.id === action.payload);
            if (algorithm) {
                algorithm.selected = !algorithm.selected;
            }
        },
    },
});

export const { setAlgorithms, toggleAlgorithm } = algorithmSlice.actions;

export const selectAlgorithms = (state: RootState) => state.algorithm.algorithms;

export default algorithmSlice.reducer;

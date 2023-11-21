import { createSlice, PayloadAction, Dispatch } from '@reduxjs/toolkit';
import { RootState, AppDispatch } from '../app/store'; // Adjust the import path to your store file

export interface DataType {
    id: number;
    name: string;
}

interface DataState {
    items: DataType[];
}

const initialState: DataState = {
    items: [],
};

export const yourDataSlice = createSlice({
    name: 'data',
    initialState,
    reducers: {
        setData: (state, action: PayloadAction<DataType[]>) => {
            state.items = action.payload;
        },
    },
});

export const { setData } = yourDataSlice.actions;

// Example thunk action
export const fetchData = () => async (dispatch: AppDispatch) => {
    // Fetch data from API or other source
    const fetchedData: DataType[] = [{ id: 1, name: 'Image 1' }, { id: 2, name: 'Image 2' }];
    dispatch(setData(fetchedData));
};

export const selectData = (state: RootState) => state.data.items;

export default yourDataSlice.reducer;

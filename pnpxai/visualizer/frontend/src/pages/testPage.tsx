// src/pages/TestPage.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { Box } from '@mui/material';

const TestPage = () => {
    return (
        <Box sx={{ display: 'flex' }}>
        <Box sx={{ p: 2, flexGrow: 1 }}>
            <Card sx={{ minWidth: 275 }}>
            <CardContent>
                <Typography variant="h5" component="div">
                Test Page
                </Typography>
                <Typography variant="body2">
                This is a test page.
                </Typography>
            </CardContent>
            </Card>
        </Box>
        </Box>
    );
    };

export default TestPage;
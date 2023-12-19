// src/components/DataDisplay.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { Box } from '@mui/material';

interface Item {
  id: number;
  name: string;
}

interface Experiment {
  explainers: Item[];
  metrics: Item[];
  name: string;
}

interface DataProps {
  data: {
    experiments: Experiment[];
  };
}



const DataDisplay: React.FC<DataProps> = ({ data }) => {
    const renderList = (items: Item[], title: string) => (
        <Box mb={2}>
            <Typography variant="h6" gutterBottom>
                {title}
            </Typography>
            {items.map((item) => (
                <Card key={item.id} sx={{ marginBottom: '10px' }}>
                    <CardContent>
                        {Object.entries(item).map(([key, value]) => (
                            <Typography key={key}>
                                {key}: {value}
                            </Typography>
                        ))}
                    </CardContent>
                </Card>
            ))}
        </Box>
    );

    return (
        <Box>
            {data?.experiments.map((experiment, index) => (
                <Box key={index}>
                    {renderList(experiment.explainers, 'Explainers')}
                    {renderList(experiment.metrics, 'Metrics')}
                </Box>
            ))}
        </Box>
    );
};

export default DataDisplay;

import React from 'react';
import { Box, Card, Grid, Typography, Divider } from '@mui/material';
import model_info from '../assets/mockup/model_info.json';

const ModelInfoPage: React.FC = () => {
  
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        
        {/* Introduction or Overview Card */}
        <Grid item xs={12}>
          <Card sx={{ p: 2 }}>
            <Typography variant="h4" gutterBottom> 모델명 : {model_info.name} </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1">
              VGG16은 이미지 인식과 분류에 널리 사용되는 딥 러닝 모델입니다.
            </Typography>
          </Card>
        </Grid>

        {/* Model Architecture */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, height: '100%' }}>
            <Typography variant="h5" gutterBottom>모델 구조</Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1">
              {model_info.structure}
            </Typography>
          </Card>
        </Grid>

        {/* Model Summary */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, height: '100%' }}>
            <Typography variant="h5" gutterBottom>모델 요약</Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1">
              {model_info.summary}
            </Typography>
          </Card>
        </Grid>

        {/* Model Characteristics */}
        <Grid item xs={12}>
          <Card sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>모델 특징</Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1">
              {model_info.characteristic}
            </Typography>
          </Card>
        </Grid>

        {/* Application Domains */}
        <Grid item xs={12}>
          <Card sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>사용 가능한 영역</Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1">
              {model_info.domain}
            </Typography>
          </Card>
        </Grid>

      </Grid>
    </Box>
  );
};

export default ModelInfoPage;

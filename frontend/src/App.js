// frontend/src/App.js
import React from 'react';
import { Container, Typography } from '@material-ui/core';
import RealTimeChart from './RealTimeChart';
import HistoricalChart from './HistoricalChart';
import StockSelector from './StockSelector';

function App() {
  return (
    <Container maxWidth="md">
      <Typography variant="h4" align="center" gutterBottom>
        Real-Time Stock Forecasting Application
      </Typography>
      <StockSelector />
      <RealTimeChart />
      <HistoricalChart />
    </Container>
  );
}

export default App;

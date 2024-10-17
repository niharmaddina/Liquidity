// frontend/src/HistoricalChart.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

function HistoricalChart() {
  const [dataPoints, setDataPoints] = useState([]);
  const [dates, setDates] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/historical-data')
      .then(response => {
        const data = response.data;
        const prices = data.map(item => item.Close);
        const dates = data.map(item => item.Date);
        setDataPoints(prices);
        setDates(dates);
      })
      .catch(error => {
        console.error('Error fetching historical data:', error);
      });
  }, []);

  const data = {
    labels: dates,
    datasets: [
      {
        label: 'Historical Closing Price',
        data: dataPoints,
        fill: false,
        backgroundColor: 'green',
        borderColor: 'green',
      },
    ],
  };

  return (
    <div>
      <h3>Historical Data</h3>
      <Line data={data} />
    </div>
  );
}

export default HistoricalChart;

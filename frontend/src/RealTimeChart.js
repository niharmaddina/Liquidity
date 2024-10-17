// frontend/src/RealTimeChart.js
import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';

function RealTimeChart() {
  const [dataPoints, setDataPoints] = useState([]);
  const [timestamps, setTimestamps] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/predictions');
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const timestamp = new Date().toLocaleTimeString();
      setDataPoints((prevData) => [...prevData, message.prediction]);
      setTimestamps((prevTimes) => [...prevTimes, timestamp]);
    };
    return () => ws.close();
  }, []);

  const data = {
    labels: timestamps,
    datasets: [
      {
        label: 'Predicted Price',
        data: dataPoints,
        fill: false,
        backgroundColor: 'blue',
        borderColor: 'blue',
      },
    ],
  };

  return (
    <div>
      <h3>Real-Time Predictions</h3>
      <Line data={data} />
    </div>
  );
}

export default RealTimeChart;

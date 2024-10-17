// frontend/src/StockSelector.js
import React, { useState } from 'react';
import { FormControl, InputLabel, Select, MenuItem } from '@material-ui/core';

function StockSelector() {
  const [symbol, setSymbol] = useState('AAPL');

  const handleChange = (event) => {
    setSymbol(event.target.value);
    // TODO: Implement symbol change logic
  };

  return (
    <FormControl fullWidth>
      <InputLabel id="stock-select-label">Stock Symbol</InputLabel>
      <Select
        labelId="stock-select-label"
        id="stock-select"
        value={symbol}
        onChange={handleChange}
      >
        <MenuItem value="AAPL">AAPL</MenuItem>
        <MenuItem value="GOOG">GOOG</MenuItem>
        <MenuItem value="MSFT">MSFT</MenuItem>
        {/* Add more stocks as needed */}
      </Select>
    </FormControl>
  );
}

export default StockSelector;

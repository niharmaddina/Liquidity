# Liquidity

A research prototype from October 2024: next-day stock-price prediction with a liquid neural
network cell followed by a neural ODE, served from a FastAPI backend over a websocket to a React
chart. Built to learn how liquid time-constant networks and `torchdiffeq` behave on noisy
financial series, not to trade with.

## What is in the repository

```
Liquidity.ipynb        research notebook: data exploration, indicators, the LNN+ODE model on TSLA
backend/
  model_definition.py  LNNCell -> ODEFunc (solved with torchdiffeq.odeint) -> linear head
  train_model.py       trains on daily AAPL closes since 2010, saves lnn_ode_model.pth
  model_inference.py   loads the weights, predicts from one feature row
  data_processing.py   yfinance download and feature construction
  main.py              FastAPI: /historical-data, /predict, websocket /ws/predictions
  lnn_ode_model.pth    trained weights (committed; 64-unit hidden state)
frontend/              React app: ticker selector, historical chart, live prediction chart
docker-compose.yml     backend on :8000, frontend on :3000
```

## The model

Input is a four-feature row per day: close, 20-day and 50-day moving averages, and daily return,
standardised with scikit-learn's `StandardScaler`. The target is the next day's close.

1. **Liquid cell.** `h = h0 + (-lambda * h0 + tanh(W x + U h0 + b))`, with a learned per-unit
   time constant `lambda`. This is one explicit Euler step of a liquid time-constant unit from a
   zero initial state.
2. **Neural ODE.** The cell's state is integrated from t=0 to t=1 by `torchdiffeq.odeint` with a
   two-layer tanh MLP as the vector field.
3. **Head.** A linear layer maps the final state to one value.

Training: MSE loss, Adam at 1e-3 with weight decay 1e-4, batch size 128, 50 epochs, hidden size
64.

## Results from the notebook (TSLA, ten years of daily data, run on Colab, October 2024)

| Metric | Validation set | All data |
|---|---|---|
| RMSE (USD) | 21.75 | 14.24 |
| MAE (USD) | 18.31 | 9.78 |
| R squared | 0.669 | 0.984 |

The "all data" column includes the training period and is reported only because the notebook
prints it; the validation column is the honest number. Training loss fell from 0.090 to the
low 0.01s over 50 epochs with validation loss tracking it. There is no comparison against a
naive baseline (predict today's close), which for next-day prices usually scores close to this,
so these numbers do not demonstrate predictive value beyond persistence.

## Running it

With Docker:

```bash
docker compose up --build        # backend http://localhost:8000, frontend http://localhost:3000
```

Without Docker:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py            # optional: retrains on AAPL and overwrites lnn_ode_model.pth
uvicorn main:app --reload        # GET /historical-data, GET /predict, WS /ws/predictions
```

```bash
cd frontend && npm install && npm start
```

## State of the code (updated 2026-10-11)

The 2024 prototype had three defects that stopped the backend from working from a fresh clone, all
fixed in the October 2026 commit:

- the committed `lnn_ode_model.pth` was an empty file; it is now real weights from `train_model.py`
  (50 epochs on AAPL, final training MSE 0.00078 in standardised units) with the fitted scalers in
  `scalers.joblib`;
- serving built features from one day of one-minute bars while training used daily bars, and fed
  them unscaled; `/predict` now builds the same daily feature row as training, scales it, and
  inverts the target scaling so the response is a price;
- `yfinance` now returns multi-level columns and pandas dropped `fillna(method=...)`; both handled.

`/predict` and the websocket accept `?symbol=` (default `AAPL`). The frontend selector is still not
wired to it, and the training ticker is still hard-coded. There is no comparison against a
persistence baseline, so treat the model as a learning exercise.

## Disclaimer

Educational project. Nothing here is financial advice or suitable for trading.

## License

MIT, see `LICENSE`.

# Walk-forward evaluation: does the LNN+ODE model beat persistence?

Run 2026-10-22 01:46 (UTC) with `python evaluate.py` from `backend/`.

Training window: targets on or before 2022-12-31 (daily bars since 2010-03-16). Test window: forecasts made on each trading day from 2023-01-03 to 2024-12-31 for the next day's close, 502 days per ticker. The fixed variant never refits; the refit variant refits from scratch every 63 trading days on all data available at that point (7 refits per ticker). Seed 0, 50 epochs, hidden size 64, torch 2.14.1, cpu (arm64). Scalers fitted on training rows only. Prices are yfinance auto-adjusted closes downloaded 2026-10-22.

## Summary (RMSE in USD, with RMSE as a percentage of the mean test-window price)

| Ticker | Mean price | Persistence (today's close) | 5-day moving average | Linear regression, fixed | LNN+ODE, fixed | LNN+ODE, refit | Dir. acc. LNN+ODE fixed / refit | Share of up days |
|---|---|---|---|---|---|---|---|---|
| AAPL | 187.97 | 2.54 (1.35%) | 3.93 (2.09%) | 2.54 (1.35%) | 7.03 (3.74%) | 3.95 (2.10%) | 48.8% / 46.0% | 56.2% |
| TSLA | 224.61 | 8.69 (3.87%) | 13.31 (5.92%) | 8.69 (3.87%) | 10.50 (4.68%) | 11.18 (4.98%) | 49.6% / 52.0% | 52.6% |
| MSFT | 360.20 | 4.94 (1.37%) | 6.98 (1.94%) | 4.94 (1.37%) | 15.96 (4.43%) | 6.89 (1.91%) | 49.4% / 47.8% | 54.4% |
| SPY | 469.33 | 3.79 (0.81%) | 5.75 (1.23%) | 3.80 (0.81%) | 15.40 (3.28%) | 5.97 (1.27%) | 47.4% / 45.6% | 57.4% |

Lower RMSE is better. "Dir. acc." is the share of test days on which the forecast called the sign of the next-day move correctly; persistence makes no call, so it has none. "Share of up days" is what always predicting "up" would have scored.

## Per-ticker detail

### AAPL

| Method | RMSE (USD) | RMSE % of mean price | MAE (USD) | Directional accuracy | Bias (mean forecast - actual, USD) |
|---|---|---|---|---|---|
| Persistence (today's close) | 2.54 | 1.35% | 1.88 | n/a | -0.24 |
| 5-day moving average | 3.93 | 2.09% | 3.10 | 46.4% | -0.73 |
| Linear regression, fixed | 2.54 | 1.35% | 1.89 | 48.8% | -0.22 |
| Linear regression, refit | 2.55 | 1.35% | 1.88 | 53.4% | -0.15 |
| LNN+ODE, fixed | 7.03 | 3.74% | 4.74 | 48.8% | -3.17 |
| LNN+ODE, refit | 3.95 | 2.10% | 3.00 | 46.0% | -1.55 |

### TSLA

| Method | RMSE (USD) | RMSE % of mean price | MAE (USD) | Directional accuracy | Bias (mean forecast - actual, USD) |
|---|---|---|---|---|---|
| Persistence (today's close) | 8.69 | 3.87% | 6.11 | n/a | -0.54 |
| 5-day moving average | 13.31 | 5.92% | 9.54 | 49.8% | -1.73 |
| Linear regression, fixed | 8.69 | 3.87% | 6.11 | 53.2% | -0.66 |
| Linear regression, refit | 8.68 | 3.87% | 6.10 | 53.8% | -0.58 |
| LNN+ODE, fixed | 10.50 | 4.68% | 7.06 | 49.6% | -3.75 |
| LNN+ODE, refit | 11.18 | 4.98% | 7.50 | 52.0% | -1.55 |

### MSFT

| Method | RMSE (USD) | RMSE % of mean price | MAE (USD) | Directional accuracy | Bias (mean forecast - actual, USD) |
|---|---|---|---|---|---|
| Persistence (today's close) | 4.94 | 1.37% | 3.76 | n/a | -0.36 |
| 5-day moving average | 6.98 | 1.94% | 5.64 | 50.6% | -1.11 |
| Linear regression, fixed | 4.94 | 1.37% | 3.75 | 52.8% | -0.38 |
| Linear regression, refit | 4.95 | 1.37% | 3.74 | 53.0% | -0.17 |
| LNN+ODE, fixed | 15.96 | 4.43% | 12.67 | 49.4% | -10.80 |
| LNN+ODE, refit | 6.89 | 1.91% | 5.57 | 47.8% | -2.92 |

### SPY

| Method | RMSE (USD) | RMSE % of mean price | MAE (USD) | Directional accuracy | Bias (mean forecast - actual, USD) |
|---|---|---|---|---|---|
| Persistence (today's close) | 3.79 | 0.81% | 2.86 | n/a | -0.42 |
| 5-day moving average | 5.75 | 1.23% | 4.52 | 46.8% | -1.27 |
| Linear regression, fixed | 3.80 | 0.81% | 2.88 | 49.6% | -0.46 |
| Linear regression, refit | 3.79 | 0.81% | 2.86 | 53.8% | -0.33 |
| LNN+ODE, fixed | 15.40 | 3.28% | 10.82 | 47.4% | -7.79 |
| LNN+ODE, refit | 5.97 | 1.27% | 4.84 | 45.6% | -2.95 |

## Data

- AAPL: 4226 daily bars 2010-01-04 to 2026-10-21, downloaded 2026-10-22T01:46:17Z; 3222 training pairs, 502 test days.
- TSLA: 4104 daily bars 2010-06-29 to 2026-10-21, downloaded 2026-10-22T01:46:20Z; 3100 training pairs, 502 test days.
- MSFT: 4226 daily bars 2010-01-04 to 2026-10-21, downloaded 2026-10-22T01:46:22Z; 3222 training pairs, 502 test days.
- SPY: 4226 daily bars 2010-01-04 to 2026-10-21, downloaded 2026-10-22T01:46:25Z; 3222 training pairs, 502 test days.

Per-day forecasts are in `predictions.csv`; `walk_forward.csv` holds this table in long form. Rebuild both and the figure without retraining with `python evaluate.py --report-only`.

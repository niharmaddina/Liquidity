# backend/data_processing.py
import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# Daily bars are cached here (gitignored) so each ticker is downloaded from yfinance once.
DATA_DIR = os.environ.get(
    'LIQUIDITY_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
)

def _flat(data):
    """yfinance >= 0.2.40 returns a (field, ticker) MultiIndex for a single ticker; keep the field level."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def get_historical_data(symbol):
    data = _flat(yf.download(symbol, start='2010-01-01', interval='1d', progress=False))
    data.reset_index(inplace=True)
    return data

def get_latest_data(symbol):
    data = _flat(yf.download(symbol, period='1d', interval='1m', progress=False))
    data.reset_index(inplace=True)
    return data.tail(1)

FEATURES = ['Close', 'MA20', 'MA50', 'Return']


def add_features(data):
    """The four model inputs, computed on DAILY bars: close, 20- and 50-day moving averages, daily return."""
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    return data.dropna()


def latest_feature_row(symbol):
    """Feature row for the most recent trading day, built from daily history so it matches training."""
    data = add_features(get_historical_data(symbol))
    return data[FEATURES].iloc[-1].values.reshape(1, -1)


def preprocess_data(data):
    """Kept for compatibility: features from an already-downloaded daily frame."""
    return add_features(data)[FEATURES].iloc[-1].values.reshape(1, -1)


def load_daily(symbol, start='2010-01-01', data_dir=None, refresh=False):
    """Daily bars for `symbol` from `start` to the download date, cached as `data/<SYMBOL>.csv`.

    The first call downloads from yfinance (auto-adjusted close, the same series `train_model.py`
    uses) and writes the CSV plus a small `.meta.json` recording when it was fetched; later calls
    read the CSV. Pass `refresh=True` to re-download.
    """
    data_dir = data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'{symbol}.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=['Date'])
    data = _flat(yf.download(symbol, start=start, interval='1d', progress=False))
    if data is None or data.empty:
        raise RuntimeError(f'yfinance returned no data for {symbol}')
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.to_csv(path, index=False)
    meta = {
        'symbol': symbol,
        'downloaded_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'rows': int(len(data)),
        'first': str(data['Date'].iloc[0].date()),
        'last': str(data['Date'].iloc[-1].date()),
        'auto_adjust': True,
    }
    with open(os.path.join(data_dir, f'{symbol}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return data


def supervised_frame(data):
    """One row per trading day t: `Date`, the four features on day t, `TargetDate` and `Target`
    (the close on the next trading day). The last day has no target and is dropped."""
    feats = add_features(data)
    out = feats[['Date'] + FEATURES].copy()
    out['TargetDate'] = feats['Date'].shift(-1)
    out['Target'] = feats['Close'].shift(-1)
    return out.iloc[:-1].reset_index(drop=True)

# backend/data_processing.py
import pandas as pd
import yfinance as yf

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

# backend/data_processing.py
import pandas as pd
import yfinance as yf

def get_historical_data(symbol):
    data = yf.download(symbol, start='2010-01-01', interval='1d')
    data.reset_index(inplace=True)
    return data

def get_latest_data(symbol):
    data = yf.download(symbol, period='1d', interval='1m')
    data.reset_index(inplace=True)
    return data.tail(1)

def preprocess_data(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    data.fillna(method='bfill', inplace=True)
    features = data[['Close', 'MA20', 'MA50', 'Return']].iloc[-1]
    features = features.values.reshape(1, -1)
    return features

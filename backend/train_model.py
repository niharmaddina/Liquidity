# backend/train_model.py
"""Train the LNN+ODE model on daily AAPL bars and save weights plus scalers.

    python train_model.py                 # 50 epochs on cached AAPL data (downloads once)
    python train_model.py --refresh       # re-download first
    python train_model.py --epochs 1 --out-dir /tmp/smoke

`fit_model` is the reusable training loop; `evaluate.py` calls it for the walk-forward test.
"""
import argparse
import logging
import os
import random
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_processing import FEATURES, load_daily, supervised_frame
from model_definition import LNN_ODE_Model

HIDDEN_SIZE = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fit_model(X, y, seed=0, epochs=50, hidden_size=HIDDEN_SIZE, batch_size=128, lr=1e-3,
              weight_decay=1e-4, device=None, log=None):
    """Fit StandardScalers on (X, y) and train the model on the scaled data.

    X: (n, 4) raw feature rows, y: (n,) raw next-day closes. Returns (model, scalers) with
    scalers = {'X': StandardScaler, 'y': StandardScaler}; the model is left in eval mode on CPU.
    The scalers are fitted on exactly the rows passed in, so callers control what is "seen".
    """
    set_seed(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    Xs = torch.tensor(scaler_X.fit_transform(np.asarray(X, dtype=np.float64)), dtype=torch.float32)
    ys = torch.tensor(scaler_y.fit_transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)),
                      dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xs.to(device), ys.to(device)), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))

    model = LNN_ODE_Model(X.shape[1], hidden_size, 1).to(device)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        epoch_loss, start = 0.0, time.time()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if log:
            log(f'Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(loader):.6f}, '
                f'Time: {time.time() - start:.2f}s')
    model.eval()
    return model.cpu(), {'X': scaler_X, 'y': scaler_y}


def predict(model, scalers, X):
    """Next-day close in price units for each raw feature row in X (n, 4)."""
    xs = torch.tensor(scalers['X'].transform(np.asarray(X, dtype=np.float64)), dtype=torch.float32)
    with torch.no_grad():
        out = model(xs).numpy()
    return scalers['y'].inverse_transform(out)[:, 0]


def train_model(symbol='AAPL', epochs=50, seed=0, out_dir='.', refresh=False):
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training on: {device}')

    sf = supervised_frame(load_daily(symbol, refresh=refresh))
    logging.info(f'{symbol}: {len(sf)} rows, {sf.Date.iloc[0].date()} to {sf.Date.iloc[-1].date()}')
    model, scalers = fit_model(sf[FEATURES].values, sf['Target'].values, seed=seed,
                               epochs=epochs, device=device, log=logging.info)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'lnn_ode_model.pth'))
    joblib.dump(scalers, os.path.join(out_dir, 'scalers.joblib'))
    logging.info(f'Model training complete; saved lnn_ode_model.pth and scalers.joblib in {out_dir}.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default='AAPL')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='.', help='where to write lnn_ode_model.pth and scalers.joblib')
    p.add_argument('--refresh', action='store_true', help='re-download the daily bars instead of using data/')
    a = p.parse_args()
    train_model(a.symbol, a.epochs, a.seed, a.out_dir, a.refresh)

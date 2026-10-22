"""Unit tests for the model, feature builder, scalers and evaluation metrics. No network access."""
import os

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from data_processing import FEATURES, add_features, supervised_frame
from evaluate import (directional_accuracy, mae, metrics_table, moving_average_forecast,
                      persistence_forecast, rmse, split_frame)
from model_definition import LNN_ODE_Model, LNNCell
from model_inference import make_prediction


def synthetic_frame(n=120, start=100.0):
    """Daily bars whose close rises by exactly 1.0 per day, so every feature has a closed form."""
    dates = pd.bdate_range('2020-01-01', periods=n)
    close = start + np.arange(n, dtype=float)
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': close + 1, 'Low': close - 1,
                         'Close': close, 'Volume': 1000})


# ---------------------------------------------------------------- model

def test_model_forward_shape_and_finite():
    torch.manual_seed(0)
    model = LNN_ODE_Model(input_size=4, hidden_size=8, output_size=1)
    out = model(torch.randn(5, 4))
    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_lnn_cell_from_zero_state_is_tanh_of_input():
    torch.manual_seed(0)
    cell = LNNCell(3, 2)
    x, h0 = torch.randn(4, 3), torch.zeros(4, 2)
    # h = h0 + (-lambda * h0 + tanh(W x + U h0 + b)); with h0 = 0 only the tanh term survives
    assert torch.allclose(cell(x, h0), torch.tanh(x @ cell.W.t() + cell.bias))


def test_model_is_deterministic_for_a_fixed_seed():
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        model = LNN_ODE_Model(4, 8, 1)
        outs.append(model(torch.ones(2, 4)))
    assert torch.equal(outs[0], outs[1])


# ---------------------------------------------------------------- scalers

def test_scaler_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(150.0, 30.0, size=(60, 4))
    y = rng.normal(150.0, 30.0, size=(60, 1))
    sx, sy = StandardScaler().fit(X), StandardScaler().fit(y)
    assert np.allclose(sx.inverse_transform(sx.transform(X)), X)
    assert np.allclose(sy.inverse_transform(sy.transform(y)), y)
    assert np.allclose(sx.transform(X).mean(axis=0), 0.0) and np.allclose(sx.transform(X).std(axis=0), 1.0)


def test_make_prediction_inverts_target_scaling():
    rng = np.random.default_rng(1)
    X = rng.normal(150.0, 30.0, size=(60, 4))
    y = rng.normal(150.0, 30.0, size=(60, 1))
    scalers = {'X': StandardScaler().fit(X), 'y': StandardScaler().fit(y)}

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    pred = make_prediction(ZeroModel(), X[:1], scalers)
    assert isinstance(pred, float)
    assert pred == pytest.approx(float(y.mean()), abs=1e-4)  # a 0 in scaled units is the training mean


# ---------------------------------------------------------------- features

def test_add_features_on_synthetic_frame():
    feats = add_features(synthetic_frame(120))
    assert set(FEATURES) <= set(feats.columns)
    assert len(feats) == 120 - 49  # the first 49 rows have no 50-day mean
    assert not feats[FEATURES].isna().any().any()
    # a close rising 1.0/day: the 20-day mean lags by 9.5, the 50-day mean by 24.5
    assert np.allclose(feats['Close'] - feats['MA20'], 9.5)
    assert np.allclose(feats['Close'] - feats['MA50'], 24.5)
    assert np.allclose(feats['Return'], 1.0 / (feats['Close'] - 1.0))


def test_add_features_does_not_modify_input():
    df = synthetic_frame(60)
    cols = list(df.columns)
    add_features(df)
    assert list(df.columns) == cols


def test_supervised_frame_target_is_next_close():
    sf = supervised_frame(synthetic_frame(120))
    assert len(sf) == 120 - 49 - 1  # last feature day has no next close
    assert list(sf.columns) == ['Date'] + FEATURES + ['TargetDate', 'Target']
    assert np.allclose(sf['Target'], sf['Close'] + 1.0)
    assert (sf['TargetDate'] > sf['Date']).all()


def test_split_frame_keeps_test_labels_out_of_training():
    sf = supervised_frame(synthetic_frame(120))
    cutoff = sf['Date'].iloc[40]
    first_test = sf['Date'].iloc[41]
    train, test = split_frame(sf, cutoff, first_test, sf['Date'].iloc[-1])
    assert (train['TargetDate'] <= cutoff).all()
    assert (test['Date'] >= first_test).all()
    assert len(train) + len(test) == len(sf) - 1  # the pair (cutoff -> next day) belongs to neither
    assert train['TargetDate'].max() < test['Date'].min()


# ---------------------------------------------------------------- metrics

CLOSE = np.array([10.0, 11.0, 12.0, 11.0, 13.0])   # today's close
ACTUAL = np.array([11.0, 12.0, 11.0, 13.0, 12.0])  # next day's close


def test_rmse_and_mae_on_toy_series():
    assert rmse(ACTUAL, ACTUAL) == 0.0 and mae(ACTUAL, ACTUAL) == 0.0
    pred = ACTUAL + np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    assert rmse(ACTUAL, pred) == pytest.approx(1.0)
    assert mae(ACTUAL, pred) == pytest.approx(1.0)
    pred = ACTUAL + np.array([3.0, 0.0, 0.0, 0.0, 4.0])
    assert rmse(ACTUAL, pred) == pytest.approx(np.sqrt(25 / 5))
    assert mae(ACTUAL, pred) == pytest.approx(7 / 5)


def test_persistence_forecast_and_its_undefined_direction():
    pred = persistence_forecast(CLOSE)
    assert np.array_equal(pred, CLOSE)
    assert rmse(ACTUAL, pred) == pytest.approx(np.sqrt(np.mean((ACTUAL - CLOSE) ** 2)))
    assert np.isnan(directional_accuracy(CLOSE, ACTUAL, pred))


def test_directional_accuracy_extremes_and_mix():
    moves = np.sign(ACTUAL - CLOSE)
    assert directional_accuracy(CLOSE, ACTUAL, CLOSE + 0.5 * moves) == 1.0
    assert directional_accuracy(CLOSE, ACTUAL, CLOSE - 0.5 * moves) == 0.0
    flipped = moves.copy()
    flipped[:2] *= -1
    assert directional_accuracy(CLOSE, ACTUAL, CLOSE + flipped) == pytest.approx(3 / 5)


def test_moving_average_forecast():
    ma = moving_average_forecast(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), window=5)
    assert np.isnan(ma[:4]).all()
    assert ma[4] == pytest.approx(3.0) and ma[5] == pytest.approx(4.0)


def test_metrics_table_matches_direct_computation():
    pred = pd.DataFrame({'Ticker': 'TOY', 'Date': pd.bdate_range('2024-01-01', periods=5),
                         'Close': CLOSE, 'Actual': ACTUAL, 'persistence': CLOSE, 'lnn_fixed': ACTUAL})
    t = metrics_table(pred).set_index('method')
    assert list(t.index) == ['persistence', 'lnn_fixed']
    assert t.loc['persistence', 'rmse'] == pytest.approx(rmse(ACTUAL, CLOSE))
    assert t.loc['persistence', 'rmse_pct'] == pytest.approx(100 * rmse(ACTUAL, CLOSE) / ACTUAL.mean())
    assert np.isnan(t.loc['persistence', 'dir_acc'])
    assert t.loc['lnn_fixed', 'rmse'] == 0.0 and t.loc['lnn_fixed', 'dir_acc'] == 1.0
    assert t.loc['lnn_fixed', 'up_share'] == pytest.approx(3 / 5)


# ---------------------------------------------------------------- committed results

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')


@pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, 'predictions.csv')), reason='results/ not present')
def test_committed_walk_forward_table_matches_predictions():
    """results/walk_forward.csv must be what metrics_table gives on results/predictions.csv, and the
    per-day file must have persistence == today's close and the target == the next row's close."""
    pred = pd.read_csv(os.path.join(RESULTS, 'predictions.csv'), parse_dates=['Date', 'TargetDate'])
    table = pd.read_csv(os.path.join(RESULTS, 'walk_forward.csv'))
    recomputed = metrics_table(pred)
    assert len(table) == len(recomputed) > 0
    for col in ['ticker', 'method', 'n']:
        assert list(table[col]) == list(recomputed[col]), col
    for col in ['mean_price', 'rmse', 'rmse_pct', 'mae', 'dir_acc', 'bias', 'up_share']:
        assert np.allclose(table[col], recomputed[col], atol=1e-5, equal_nan=True), col
    assert (pred['persistence'] == pred['Close']).all()
    for _, g in pred.groupby('Ticker'):
        assert np.allclose(g['Actual'].to_numpy()[:-1], g['Close'].to_numpy()[1:])
        assert (g['TargetDate'] > g['Date']).all()

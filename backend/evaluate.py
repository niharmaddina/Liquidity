# backend/evaluate.py
"""Walk-forward evaluation of the LNN+ODE model against naive baselines.

Train on daily bars whose target date is on or before --train-end, then forecast the next close
one step ahead for every trading day from --test-start to --test-end without refitting ("fixed").
If --refit-every is positive, a second variant refits the model and the linear regression every
that many trading days on all data available at that point ("refit"). Baselines: persistence
(predict today's close), ordinary least squares on the same four features, and the 5-day moving
average of closes. Scalers are fitted on the training rows only; seeds are fixed.

    python evaluate.py                                  # AAPL TSLA MSFT SPY, fixed + refit every 63 days
    python evaluate.py --tickers AAPL --refit-every 0   # fixed variant only
    python evaluate.py --report-only                    # rebuild tables and figure from results/predictions.csv

Outputs under --out (default ../results): predictions.csv (one row per test day and ticker),
walk_forward.csv, walk_forward.md, run_meta.json, figures/prediction_vs_actual.png.
"""
import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from data_processing import DATA_DIR, FEATURES, load_daily, supervised_frame
from train_model import fit_model, predict

DEFAULT_TICKERS = ['AAPL', 'TSLA', 'MSFT', 'SPY']
METHOD_LABELS = {
    'persistence': "Persistence (today's close)",
    'ma5': '5-day moving average',
    'linreg_fixed': 'Linear regression, fixed',
    'linreg_refit': 'Linear regression, refit',
    'lnn_fixed': 'LNN+ODE, fixed',
    'lnn_refit': 'LNN+ODE, refit',
}
METHOD_ORDER = list(METHOD_LABELS)


# ----------------------------------------------------------------------------- metrics

def rmse(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(pred - actual)))


def directional_accuracy(prev_close, actual, pred):
    """Share of days where sign(pred - prev_close) == sign(actual - prev_close).

    A forecast that never moves away from today's close (persistence) makes no directional
    call, so the result is NaN rather than 0."""
    prev_close, actual, pred = (np.asarray(a, float) for a in (prev_close, actual, pred))
    pred_dir = np.sign(pred - prev_close)
    if np.all(pred_dir == 0):
        return float('nan')
    return float(np.mean(pred_dir == np.sign(actual - prev_close)))


def persistence_forecast(close):
    """Forecast for the next close = today's close."""
    return np.asarray(close, float).copy()


def moving_average_forecast(close, window=5):
    """Forecast for the next close = mean of the last `window` closes including today's."""
    return pd.Series(np.asarray(close, float)).rolling(window).mean().to_numpy()


# ----------------------------------------------------------------------------- evaluation

def split_frame(sf, train_end, test_start, test_end):
    """Training rows: target date <= train_end (nothing after the cutoff is seen, not even as a
    label). Test rows: feature date in [test_start, test_end]; the target is the next close."""
    train = sf[sf['TargetDate'] <= pd.Timestamp(train_end)]
    test = sf[(sf['Date'] >= pd.Timestamp(test_start)) & (sf['Date'] <= pd.Timestamp(test_end))]
    return train, test


def fit_linear(X, y, scalers):
    return LinearRegression().fit(scalers['X'].transform(X), scalers['y'].transform(y.reshape(-1, 1)))


def predict_linear(lr, scalers, X):
    return scalers['y'].inverse_transform(lr.predict(scalers['X'].transform(X)))[:, 0]


def walk_forward(sf, train_end, test_start, test_end, seed=0, epochs=50, refit_every=63, log=None):
    """Return a frame with one row per test day: Date, TargetDate, Close, Actual and a column of
    forecasts per method. `sf` is a supervised frame with a RangeIndex."""
    log = log or (lambda *_: None)
    train, test = split_frame(sf, train_end, test_start, test_end)
    X_tr, y_tr = train[FEATURES].to_numpy(float), train['Target'].to_numpy(float)
    X_te = test[FEATURES].to_numpy(float)
    log(f'train {len(train)} rows ({train.Date.iloc[0].date()} to {train.Date.iloc[-1].date()}, '
        f'last target {train.TargetDate.iloc[-1].date()}); test {len(test)} rows '
        f'({test.Date.iloc[0].date()} to {test.Date.iloc[-1].date()}, '
        f'last target {test.TargetDate.iloc[-1].date()})')

    t0 = time.time()
    model, scalers = fit_model(X_tr, y_tr, seed=seed, epochs=epochs)
    log(f'fixed fit done in {time.time() - t0:.0f}s')
    lr = fit_linear(X_tr, y_tr, scalers)

    out = test[['Date', 'TargetDate', 'Close', 'Target']].rename(columns={'Target': 'Actual'}).copy()
    out['persistence'] = persistence_forecast(test['Close'].to_numpy())
    out['ma5'] = moving_average_forecast(sf['Close'].to_numpy(), 5)[test.index.to_numpy()]
    out['linreg_fixed'] = predict_linear(lr, scalers, X_te)
    out['lnn_fixed'] = predict(model, scalers, X_te)

    if refit_every and refit_every > 0:
        # Block 0 reuses the fixed fit (trained through train_end); each later block is predicted
        # by a model refit from scratch on every pair whose target is known at the block's start.
        lnn_refit, linreg_refit = out['lnn_fixed'].to_numpy().copy(), out['linreg_fixed'].to_numpy().copy()
        n = len(test)
        for start in range(refit_every, n, refit_every):
            end = min(start + refit_every, n)
            cutoff = test['Date'].iloc[start]
            hist = sf[sf['TargetDate'] <= cutoff]
            t0 = time.time()
            m, s = fit_model(hist[FEATURES].to_numpy(float), hist['Target'].to_numpy(float),
                             seed=seed, epochs=epochs)
            lnn_refit[start:end] = predict(m, s, X_te[start:end])
            linreg_refit[start:end] = predict_linear(
                fit_linear(hist[FEATURES].to_numpy(float), hist['Target'].to_numpy(float), s), s, X_te[start:end])
            log(f'refit at test row {start} (data through {cutoff.date()}, {len(hist)} rows) '
                f'in {time.time() - t0:.0f}s')
        out['linreg_refit'] = linreg_refit
        out['lnn_refit'] = lnn_refit
    return out.reset_index(drop=True)


def metrics_table(pred):
    """Long table: one row per (ticker, method) with n, mean price, RMSE, RMSE %, MAE, directional
    accuracy, bias and the share of up days in the window."""
    rows = []
    for ticker, g in pred.groupby('Ticker', sort=False):
        actual, prev = g['Actual'].to_numpy(float), g['Close'].to_numpy(float)
        mean_price = float(actual.mean())
        up_share = float(np.mean(actual > prev))
        for m in METHOD_ORDER:
            if m not in g or g[m].isna().all():
                continue
            p = g[m].to_numpy(float)
            rows.append({
                'ticker': ticker, 'method': m, 'label': METHOD_LABELS[m], 'n': int(len(g)),
                'mean_price': mean_price, 'rmse': rmse(actual, p),
                'rmse_pct': 100 * rmse(actual, p) / mean_price, 'mae': mae(actual, p),
                'dir_acc': directional_accuracy(prev, actual, p),
                'bias': float(np.mean(p - actual)), 'up_share': up_share,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reporting

def _fmt_pct(x):
    return 'n/a' if pd.isna(x) else f'{100 * x:.1f}%'


def write_markdown(table, meta, path):
    tickers = list(dict.fromkeys(table['ticker']))
    has_refit = 'lnn_refit' in set(table['method'])
    L = []
    L.append('# Walk-forward evaluation: does the LNN+ODE model beat persistence?\n')
    L.append(f"Run {meta['run_date_utc']} (UTC) with `{meta['command']}` from `backend/`.\n")
    L.append(f"Training window: targets on or before {meta['train_end']} (daily bars since {meta['data_start']}). "
             f"Test window: forecasts made on each trading day from {meta['test_start']} to {meta['test_end']} "
             f"for the next day's close, {meta['n_test']} days per ticker. The fixed variant never refits; "
             + (f"the refit variant refits from scratch every {meta['refit_every']} trading days on all data "
                f"available at that point ({meta['n_refits']} refits per ticker). " if has_refit else '')
             + f"Seed {meta['seed']}, {meta['epochs']} epochs, hidden size {meta['hidden_size']}, "
             f"torch {meta['torch']}, {meta['device']}. Scalers fitted on training rows only. "
             f"Prices are yfinance auto-adjusted closes downloaded {meta['data_downloaded_utc']}.\n")

    L.append('## Summary (RMSE in USD, with RMSE as a percentage of the mean test-window price)\n')
    cols = ['persistence', 'ma5', 'linreg_fixed', 'lnn_fixed'] + (['lnn_refit'] if has_refit else [])
    head = '| Ticker | Mean price | ' + ' | '.join(METHOD_LABELS[c] for c in cols) + ' | Dir. acc. LNN+ODE fixed' + (' / refit' if has_refit else '') + ' | Share of up days |'
    L.append(head)
    L.append('|' + '---|' * (head.count('|') - 1))
    for t in tickers:
        g = table[table.ticker == t].set_index('method')
        cells = [f"{g.loc[c, 'rmse']:.2f} ({g.loc[c, 'rmse_pct']:.2f}%)" for c in cols]
        da = _fmt_pct(g.loc['lnn_fixed', 'dir_acc']) + (f" / {_fmt_pct(g.loc['lnn_refit', 'dir_acc'])}" if has_refit else '')
        L.append(f"| {t} | {g.loc['persistence', 'mean_price']:.2f} | " + ' | '.join(cells)
                 + f" | {da} | {_fmt_pct(g.loc['persistence', 'up_share'])} |")
    L.append('')
    L.append('Lower RMSE is better. "Dir. acc." is the share of test days on which the forecast called the '
             'sign of the next-day move correctly; persistence makes no call, so it has none. "Share of up days" '
             'is what always predicting "up" would have scored.\n')

    L.append('## Per-ticker detail\n')
    for t in tickers:
        g = table[table.ticker == t]
        L.append(f'### {t}\n')
        L.append('| Method | RMSE (USD) | RMSE % of mean price | MAE (USD) | Directional accuracy | Bias (mean forecast - actual, USD) |')
        L.append('|---|---|---|---|---|---|')
        for _, r in g.iterrows():
            L.append(f"| {r['label']} | {r['rmse']:.2f} | {r['rmse_pct']:.2f}% | {r['mae']:.2f} | "
                     f"{_fmt_pct(r['dir_acc'])} | {r['bias']:+.2f} |")
        L.append('')

    L.append('## Data\n')
    for t in tickers:
        d = meta['data'][t]
        L.append(f"- {t}: {d['rows']} daily bars {d['first']} to {d['last']}, downloaded {d['downloaded_at_utc']}; "
                 f"{d['train_rows']} training pairs, {d['test_rows']} test days.")
    L.append('')
    L.append('Per-day forecasts are in `predictions.csv`; `walk_forward.csv` holds this table in long form. '
             'Rebuild both and the figure without retraining with `python evaluate.py --report-only`.')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def plot_figure(pred, ticker, path, train_end='the training window'):
    """Top: the test window for one ticker, actual next-day close vs the fixed LNN+ODE forecast vs
    persistence. Bottom: RMSE by calendar month for the same forecasts (plus the refit variant)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    C_LNN, C_PERS, C_REFIT = '#2a78d6', '#eb6834', '#1baf7a'  # categorical slots 1-3 (validated)
    g = pred[pred.Ticker == ticker].copy()
    has_refit = 'lnn_refit' in g and g['lnn_refit'].notna().any()

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': AXIS, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), height_ratios=[3, 1.6], constrained_layout=True)

    x = g['TargetDate']
    ax.plot(x, g['Actual'], color=INK2, lw=1.6, label='Actual close', zorder=3)
    ax.plot(x, g['persistence'], color=C_PERS, lw=1.6, label="Persistence (today's close)", zorder=4)
    ax.plot(x, g['lnn_fixed'], color=C_LNN, lw=1.6, label=f'LNN+ODE, fixed (trained once, through {train_end})', zorder=5)
    for key, col, txt in (('Actual', INK2, 'actual'), ('persistence', C_PERS, 'persistence'), ('lnn_fixed', C_LNN, 'LNN+ODE fixed')):
        ax.annotate(txt, (x.iloc[-1], g[key].iloc[-1]), xytext=(6, 0), textcoords='offset points',
                    color=col, fontsize=9, va='center')
    ax.set_title(f'{ticker}: next-day close forecasts over the test window '
                 f'({g.Date.iloc[0].date()} to {g.Date.iloc[-1].date()})', loc='left', color=INK, fontsize=12)
    ax.set_ylabel('Close (USD, adjusted)')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.margins(x=0.01)
    ax.set_xlim(x.iloc[0], x.iloc[-1] + pd.Timedelta(days=40))

    month = g['Date'].dt.to_period('M')
    series = [('persistence', 'Persistence', C_PERS), ('lnn_fixed', 'LNN+ODE fixed', C_LNN)]
    if has_refit:
        series.append(('lnn_refit', 'LNN+ODE refit', C_REFIT))
    monthly = g.groupby(month).apply(lambda d: pd.Series({k: rmse(d['Actual'], d[k]) for k, _, _ in series}), include_groups=False)
    idx = np.arange(len(monthly))
    width = 0.8 / len(series)
    for i, (k, lab, col) in enumerate(series):
        ax2.bar(idx + (i - (len(series) - 1) / 2) * width, monthly[k], width=width * 0.92, color=col, label=lab, zorder=3)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([p.strftime('%b %y') for p in monthly.index], rotation=0, fontsize=8)
    ax2.set_ylabel('RMSE (USD)')
    ax2.set_title('Error by month: RMSE of each forecast against the actual next-day close', loc='left', color=INK, fontsize=11)
    ax2.legend(frameon=False, loc='upper left', fontsize=9, ncol=len(series))
    ax2.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)
    ax2.margins(x=0.01)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_outputs(pred, meta, out_dir):
    table = metrics_table(pred)
    table.to_csv(os.path.join(out_dir, 'walk_forward.csv'), index=False, float_format='%.6f')
    write_markdown(table, meta, os.path.join(out_dir, 'walk_forward.md'))
    plot_figure(pred, meta['figure_ticker'], os.path.join(out_dir, 'figures', 'prediction_vs_actual.png'),
                train_end=meta['train_end'])
    return table


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS)
    p.add_argument('--train-end', default='2022-12-31')
    p.add_argument('--test-start', default='2023-01-03')
    p.add_argument('--test-end', default='2024-12-31')
    p.add_argument('--refit-every', type=int, default=63, help='trading days between refits; 0 disables the refit variant')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--figure-ticker', default=None, help='ticker drawn in the figure (default: first ticker)')
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
    p.add_argument('--report-only', action='store_true', help='rebuild tables and figure from an existing predictions.csv')
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger('evaluate').info
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    pred_path, meta_path = os.path.join(out_dir, 'predictions.csv'), os.path.join(out_dir, 'run_meta.json')

    if a.report_only:
        pred = pd.read_csv(pred_path, parse_dates=['Date', 'TargetDate'])
        with open(meta_path) as f:
            meta = json.load(f)
        table = write_outputs(pred, meta, out_dir)
        print(table.to_string(index=False))
        return

    import sys
    meta = {
        'run_date_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
        'command': 'python evaluate.py ' + ' '.join(sys.argv[1:]) if sys.argv[1:] else 'python evaluate.py',
        'train_end': a.train_end, 'test_start': a.test_start, 'test_end': a.test_end,
        'refit_every': a.refit_every, 'seed': a.seed, 'epochs': a.epochs, 'hidden_size': 64,
        'torch': torch.__version__, 'device': 'cuda' if torch.cuda.is_available() else f'cpu ({platform.machine()})',
        'figure_ticker': a.figure_ticker or a.tickers[0], 'data': {},
    }
    frames = []
    for ticker in a.tickers:
        raw = load_daily(ticker)
        with open(os.path.join(DATA_DIR, f'{ticker}.meta.json')) as f:
            dmeta = json.load(f)
        sf = supervised_frame(raw)
        log(f'== {ticker}: {len(sf)} supervised rows')
        res = walk_forward(sf, a.train_end, a.test_start, a.test_end, seed=a.seed, epochs=a.epochs,
                           refit_every=a.refit_every, log=lambda m: log(f'{ticker}: {m}'))
        res.insert(0, 'Ticker', ticker)
        frames.append(res)
        train, _ = split_frame(sf, a.train_end, a.test_start, a.test_end)
        dmeta.update({'train_rows': int(len(train)), 'test_rows': int(len(res))})
        meta['data'][ticker] = dmeta
        meta['n_test'] = int(len(res))
        meta['n_refits'] = len(range(a.refit_every, len(res), a.refit_every)) if a.refit_every > 0 else 0
        meta['data_start'] = str(sf.Date.iloc[0].date())
        meta['data_downloaded_utc'] = dmeta['downloaded_at_utc'][:10]
        # save after every ticker so a long run can be inspected or resumed in --report-only mode
        pd.concat(frames).to_csv(pred_path, index=False, float_format='%.4f')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        log(f'{ticker}: saved {pred_path}')

    pred = pd.concat(frames, ignore_index=True)
    table = write_outputs(pred, meta, out_dir)
    print(table[['ticker', 'label', 'rmse', 'rmse_pct', 'mae', 'dir_acc']].to_string(index=False))


if __name__ == '__main__':
    main()

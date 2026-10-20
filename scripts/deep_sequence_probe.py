"""Can a sequence deep-learning model find what the 8 summary features miss? (research probe, never used live)

Instead of hand-made features, a small 1D CNN reads the raw last-60-session path of each evaluation:
daily log return, high/low vs close, opening gap, volume vs its 50-day average, and Nifty's daily return.
Two variants: "seq" (path only) and "hybrid" (path + all tabular FEATURES).

Judged by EXACTLY the production protocol (app.research.meta): same walk-forward folds and embargo, same
cut (median score on the training signals), same pre-registered adoption checks, on the untouched hold-out
and on unseen stocks. The variant that counts is the one chosen on the SELECTION period; the other's hold-out
row is printed for transparency only (choosing by hold-out would be snooping).

Leakage guards: windows end at the evaluation date's close (the trade enters at the next open); scaling and
early stopping use only the training fold (early-stopping split = its latest 15% by date).

Usage: .venv/bin/python -m scripts.deep_sequence_probe
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from app import data
from app.research import ledger, meta
from app.research.features import FEATURES
from app.universe import BENCHMARK

LOOKBACK = 60
CHANNELS = ("ret", "high", "low", "gap", "volume", "nifty")
SEEDS = (0, 1, 2)                 # small ensemble: one seed's luck must not decide the verdict
MAX_EPOCHS, PATIENCE, BATCH = 30, 4, 512
VALID_SHARE = 0.15
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


# ---------- inputs ----------

def channel_matrix(frame: pd.DataFrame, nifty: pd.Series) -> np.ndarray:
    close = frame["Close"]
    volume = np.log((frame["Volume"] + 1) / (frame["Volume"].rolling(50).mean() + 1))
    cols = [np.log(close / close.shift()), np.log(frame["High"] / close), np.log(frame["Low"] / close),
            np.log(frame["Open"] / close.shift()), volume,
            np.log(nifty.reindex(frame.index).ffill()).diff()]
    out = np.column_stack([c.to_numpy(dtype="float64") for c in cols])
    out[:, :4] = np.clip(out[:, :4], -0.2, 0.2)
    out[:, 4] = np.clip(out[:, 4], -3, 3)
    out[:, 5] = np.clip(out[:, 5], -0.2, 0.2)
    return np.nan_to_num(out)


def build_windows(df: pd.DataFrame) -> np.ndarray:
    """(rows, channels, LOOKBACK) aligned with df.index; each window ends AT the evaluation date."""
    nifty = data.load_history(BENCHMARK)["Close"]
    out = np.zeros((len(df), len(CHANNELS), LOOKBACK), dtype="float32")
    for symbol, rows in df.groupby("symbol"):
        frame = data.load_history(symbol)
        if frame is None:
            continue
        m = channel_matrix(frame, nifty)
        padded = np.vstack([np.zeros((LOOKBACK, m.shape[1])), m])
        views = np.lib.stride_tricks.sliding_window_view(padded, (LOOKBACK, m.shape[1]))[:, 0]   # (T+1, L, C)
        pos = frame.index.searchsorted(rows["date"].to_numpy(), side="right") - 1                 # last bar <= date
        out[rows.index.to_numpy()] = views[pos + 1].transpose(0, 2, 1)
    return out


# ---------- model ----------

class Net(nn.Module):
    def __init__(self, n_tabular: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(len(CHANNELS), 16, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32 + n_tabular, 1))

    def forward(self, seq, tab):
        h = self.conv(seq)
        return self.head(torch.cat([h.mean(-1), h.amax(-1), tab], dim=1)).squeeze(1)


def _train_one(seed, xs, xt, y, valid):
    torch.manual_seed(seed)
    net = Net(xt.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    fit_idx, val_idx = np.where(~valid)[0], np.where(valid)[0]
    best, best_state, stale = np.inf, None, 0
    gen = np.random.default_rng(seed)
    for _ in range(MAX_EPOCHS):
        net.train()
        for batch in np.array_split(gen.permutation(fit_idx), max(1, len(fit_idx) // BATCH)):
            opt.zero_grad()
            loss_fn(net(xs[batch], xt[batch]), y[batch]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(xs[val_idx], xt[val_idx]), y[val_idx]).item()
        if val < best - 1e-4:
            best, best_state, stale = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        elif (stale := stale + 1) >= PATIENCE:
            break
    net.load_state_dict(best_state)
    return net.eval()


def make_fit(windows: np.ndarray):
    """A meta.oos_scores-compatible fit function. `factory` is the variant name: 'seq' or 'hybrid'."""
    def fit(variant: str, train: pd.DataFrame, test: pd.DataFrame):
        seq_mean = windows[train.index].mean(axis=(0, 2), keepdims=True)
        seq_std = windows[train.index].std(axis=(0, 2), keepdims=True) + 1e-8
        cols = list(FEATURES) if variant == "hybrid" else []
        med = train[cols].median()
        tab_mean, tab_std = train[cols].fillna(med).mean(), train[cols].fillna(med).std().replace(0, 1)

        def tensors(rows):
            xs = torch.tensor((windows[rows.index] - seq_mean) / seq_std, dtype=torch.float32)
            tab = ((rows[cols].fillna(med) - tab_mean) / tab_std).to_numpy(dtype="float32") if cols \
                else np.zeros((len(rows), 0), dtype="float32")
            return xs, torch.tensor(tab)

        ordered = train.sort_values("date")
        valid = (ordered["date"] >= ordered["date"].quantile(1 - VALID_SHARE)).to_numpy()
        xs, xt = tensors(ordered)
        y = torch.tensor(ordered["label"].to_numpy(dtype="float32"))
        nets = [_train_one(seed, xs, xt, y, valid) for seed in SEEDS]

        def score(rows):
            a, b = tensors(rows)
            with torch.no_grad():
                return np.mean([torch.sigmoid(n(a, b)).numpy() for n in nets], axis=0)

        signals = train[train["buy"]]
        cut = float(np.median(score(signals))) if len(signals) else 0.5
        return score(test), cut, cols
    return fit


# ---------- report ----------

def row(label: str, result: dict) -> str:
    f = result.get("filtered", {})
    ok, failed = meta.verdict(result)
    return (f"  {label:34}{f.get('n', 0):>6}{f.get('avg_return', float('nan')):>+9.2%}{f.get('edge', float('nan')):>+8.2%}"
            f"{f.get('t_stat', float('nan')):>7.2f}{result.get('folds_positive', float('nan')):>8.0%}  "
            f"{'PASS' if ok else 'FAIL'}")


def main() -> None:
    started = time.time()
    raw, _ = ledger.load()
    df = meta.prepare(raw)
    windows = build_windows(df)
    folds = meta.folds_for(df)
    selection = [f for f in folds if f[1] < meta.HOLDOUT_START]
    holdout = [f for f in folds if f[1] >= meta.HOLDOUT_START]
    fit = make_fit(windows)
    print(f"{len(df)} rows, windows {windows.shape}, {len(selection)} selection + {len(holdout)} hold-out folds")
    print(f"  {'':34}{'trades':>6}{'avg':>9}{'edge':>8}{'t':>7}{'folds+':>8}  (pass: t >= {meta.MIN_T}, "
          f"folds+ >= {meta.FOLDS_POSITIVE:.0%}, beats rule)")

    base = meta.score_filter(meta.oos_scores(df, meta.candidates()["logistic"], holdout))
    print(row("hold-out: unfiltered rule", {**base, "filtered": base["rule"]}))
    print(row("hold-out: logistic (production)", base))

    chosen = {}
    for variant in ("seq", "hybrid"):
        chosen[variant] = meta.score_filter(meta.oos_scores(df, variant, selection, fit=fit))
        print(row(f"selection: {variant} CNN", chosen[variant]))
    best = max(chosen, key=lambda v: np.nan_to_num(chosen[v].get("filtered", {}).get("t_stat", -9), nan=-9))
    for variant in ("seq", "hybrid"):
        tag = "CHOSEN" if variant == best else "info only"
        print(row(f"hold-out: {variant} CNN ({tag})", meta.score_filter(meta.oos_scores(df, variant, holdout, fit=fit))))
    print(row(f"hold-out unseen stocks: {best} CNN",
              meta.score_filter(meta.oos_scores(df, best, holdout, symbol_holdout=True, fit=fit))))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

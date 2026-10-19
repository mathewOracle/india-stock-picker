# India Stock Picker

Three-pillar screener for ~63 large-cap NSE stocks: **fundamental + technical + chart analysis**,
combined through hard decision gates into Strong Buy / Buy / Watchlist / Hold / Avoid, each with a
concrete trade plan. Data from Yahoo Finance, cached in SQLite. **Not investment advice.**

## Run

```bash
uv venv && uv pip install -e . --group dev   # add the Walmart index flags on the corp network
./serve.sh                                     # http://127.0.0.1:61287 (sets the corp proxy for you)
```

Click **Refresh data**: it runs in the background with live progress (~35s) and reloads when done.
If Yahoo is unreachable you get a visible error and the previous data stays put.

## Architecture (`app/`)

| Module | Job |
|---|---|
| `data.py` | Yahoo fetch (fresh HTTP session per refresh, retries) + SQLite cache |
| `jobs.py` | Background refresh job with progress/failed/done status |
| `indicators.py` | Pure math: EMA/SMA, RSI, MACD, Bollinger, ATR, ADX, Stochastic, OBV, Supertrend |
| `technical.py` | Technical pillar: 13 checks incl. relative strength vs Nifty and extension vs 20-EMA |
| `patterns.py` | Chart pillar: swing structure, channel, 52w breakout, S/R (incl. 52w high/low), squeeze, pullback, double top/bottom, candlesticks, RSI divergence |
| `fundamentals.py` | Fundamental pillar from 4-5y statements: ROE, CAGRs, margins, leverage, cash conversion, FCF, Piotroski, P/E vs sector, PEG, analysts |
| `tradeplan.py` | Entry / stop (support or ATR) / targets (reachable resistance, else R-multiples) |
| `scoring.py` | Composite (40/35/25) + gates + red flags -> verdict |
| `analysis.py` | Orchestration, market regime, sector table, chart payload |

Every check emits a `Signal` (points -1..+1, weight, plain-English note); a pillar score is
`50 + 50 * weighted average`. Adding a check = adding one function.

## Decision rules

- **Buy**: composite >= 57, fundamental >= 52, technical >= 52, chart >= 45, reward/risk >= 1.5, no hard red flag.
- **Strong Buy**: 66 / 60 / 62 / 55 / 2.0, and not extended (> 3 ATR above the 20-EMA = chasing).
- **Watchlist**: fundamental >= 72 and composite >= 50, but a Buy gate failed (shown as "waiting for").
- Hard red flags: falling knife, loss-making, high leverage, weak Piotroski (non-financials only).

## Tests

```bash
.venv/bin/python -m pytest
```

## Known limits

Not backtested. Large caps only. No news/earnings-date awareness. Yahoo data can lag or be wrong
(e.g. corporate actions such as the Tata Motors demerger).

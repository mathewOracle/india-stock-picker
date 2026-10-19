# India Stock Picker

Explainable, rules-based screener for ~63 large-cap NSE stocks. Data from Yahoo Finance,
cached in SQLite. **Not investment advice.**

## Run

```bash
uv venv && uv pip install -e . --group dev   # add the Walmart index flags on the corp network
export HTTPS_PROXY=http://sysproxy.wal-mart.com:8080 HTTP_PROXY=$HTTPS_PROXY   # corp network only
.venv/bin/uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and click **Fetch market data** (30-60s on first run).

## How the score works (`app/scoring.py`)

Each metric is percentile-ranked inside the universe, averaged into six factors, then weighted:

| Factor    | Weight | Metrics                                         |
|-----------|--------|-------------------------------------------------|
| Momentum  | 25%    | 3M, 6M and 12-1M returns                        |
| Quality   | 25%    | ROE, net margin, debt/equity (not for financials)|
| Growth    | 15%    | revenue and earnings growth                      |
| Valuation | 15%    | P/E, P/B (lower is better)                       |
| Trend     | 10%    | vs 200-DMA, distance from 52w high, 50/200 cross |
| Risk      | 10%    | volatility, max drawdown                         |

Guardrails: no Buy for loss-makers or falling knives; no Strong Buy below the 200-DMA.

## Tests

```bash
.venv/bin/python -m pytest
```

## Ideas (not built, on purpose)

Backtesting the score, sector-relative valuation, mid/small caps, watchlists.

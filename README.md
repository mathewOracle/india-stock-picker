# India Stock Picker

Two tools in one app for ~63 large-cap NSE stocks:

* **Swing & positional** - fundamental + technical + chart analysis, combined through hard decision
  gates into Strong Buy / Buy / Watchlist / Hold / Avoid, each with a trade plan.
* **Intraday** - 5-minute scanner with VWAP, opening range, CPR/pivots, relative volume and a
  7-setup playbook; liquidity/volatility/results-day filters; long/short trade plans and position sizing.

Data from Yahoo Finance (may lag a few minutes), cached in SQLite. **Not investment advice.**

## Run

```bash
uv venv && uv pip install -e . --group dev   # add the Walmart index flags on the corp network
./serve.sh                                     # http://127.0.0.1:61287 (sets the corp proxy for you)
```

Refresh buttons run in the background with live progress and reload when done; failures show a
visible error and keep the previous data. Daily refresh takes ~9s (fundamentals are re-fetched only
once every 20h); intraday refresh ~8s.

## Performance

| | Before | After |
|---|---|---|
| Daily refresh | 35s (all statements every time) | ~9s (batched prices, fundamentals cached 20h) |
| CSS | 450 KB JS compiling Tailwind in the browser | 14.6 KB precompiled (`scripts/build_css.sh`) |
| Stock page transfer | 169 KB | 27 KB (gzip) |
| Analysis | memoised per data refresh (~0.4s cold, ~1ms warm) | unchanged: measured, not a bottleneck |

After changing Tailwind classes in templates, run `./scripts/build_css.sh` (needs Node; the built
CSS is committed so running the app does not).

## Intraday (`app/intraday/`)

| Module | Job |
|---|---|
| `indicators.py` | VWAP + sigma bands, opening range, floor pivots + CPR, relative volume by time of day |
| `snapshot.py` | Immutable "where is it now" view shared by checks and setups |
| `checks.py` | Technical signals (today's tape) + context signals (liquidity, ATR%, results, daily trend, quality, sector) |
| `setups.py` | Playbook: ORB, VWAP pullback, gap-and-go, gap fill, CPR trend day, PDH/PDL break, VWAP reversion |
| `analysis.py` | Market phase (IST clock), verdicts with a reason for every "No trade", chart payload |

Verdict = playbook setup + bias agreement (long >= 55, short <= 45; fades exempt) + R:R >= 1.5 +
tradeable (liquid, >= 1% ATR, not results day) + time rules (wait until 09:30, no entries after 14:45).

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

## Research & validation (`app/research/`)

No Buy is final until the evidence backs it. `Run research` on the Research page (or
`python -m app.research all`) does three things, in order:

1. **Backtest** (`backtest.py`, `evidence.py`): replays the live price rules weekly over 10 years for all
   stocks. No lookahead (proved by a test that cuts off the future), entry at next open, stop-first on
   ambiguous days, 0.15%/side costs, compared with a control group of all evaluations, with a
   pre-registered pass rule (>= 30 trades, t >= 1.65). Fundamentals are NOT backtested (Yahoo keeps
   only 4-5 years of statements).
2. **News sentiment** (`news.py`, `sentiment.py`): Google News RSS (last 14 days) with alias-based
   relevance filtering, dedupe, and a FinBERT *classifier* (positive/negative/neutral only, so it cannot
   invent text). Falls back to a labelled word list if torch/transformers are missing
   (`uv pip install -e ".[finbert]"`). Needs 5+ headlines from 3+ sources or it says "insufficient".
   News can't be backtested, so it only vetoes (score < -0.30), never proves.
3. **ML model** (`ml.py`): logistic regression vs gradient boosting vs random forest, chosen by
   walk-forward out-of-sample AUC (expanding window, 40-day embargo), against a coin-flip and a
   rules-only baseline. It only influences verdicts if it clearly beats them (`adopt()`).

`verdict.py` combines these: a rule-based Buy that fails validation is shown as **Watchlist** (with the
failed checks listed as "waiting for"), and appears under "Rule-based candidates (not validated)".

**Current finding (19 Oct 2026):** 2,929 backtested Buy trades, edge vs random +0.04% (t = 0.33): no
proven edge. No ML model beat chance out-of-sample (AUC about 0.50). So nothing is currently validated
as a Buy. That is the system working, not failing.

`python -m scripts.backtest_sanity` checks the engine can detect a planted edge (an oracle) and prints
per-feature information coefficients.

## Tests

```bash
.venv/bin/python -m pytest          # 109 tests, no network needed (ML tests take ~25s)
node tests/js/position-size.test.js
```

## Known limits

Not backtested. Large caps only. No news/earnings-date awareness. Yahoo data can lag or be wrong
(e.g. corporate actions such as the Tata Motors demerger).

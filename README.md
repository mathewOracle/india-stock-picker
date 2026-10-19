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

**Everything is incremental and permanent.** The 10-year backtest lives in an append-only ledger
(`data/research/ledger.csv.gz`); price history is *merged* onto what is stored (never a sliding window,
with dividend/split restatements rescaled); the news archive is per stock and per month. Later runs only
compute new dates, settle still-open trades, and re-fetch the current month. If Yahoo restates one
stock's prices, only that stock is replayed. `python -m scripts.verify_ledger` proves the ledger still
reproduces the archived baseline (`data/research/archive/baseline-2026-10-19/`, with SHA-256 sums).

CLI steps: `python -m app.research backtest | news | news_history | study | model | all`.

No Buy is final until the evidence backs it. `Run research` on the Research page (or
`python -m app.research all`) does three things, in order:

1. **Backtest** (`backtest.py`, `evidence.py`): replays the live price rules weekly over 10 years for all
   stocks. No lookahead (proved by a test that cuts off the future), entry at next open, stop-first on
   ambiguous days, 0.15%/side costs, compared with a control group of all evaluations, with a
   pre-registered pass rule (>= 30 trades, t >= 1.65). Fundamentals are NOT backtested (Yahoo keeps
   only 4-5 years of statements).
2. **News sentiment** (`news.py`, `sentiment.py`): Google News RSS (last 14 days) with alias-based
   relevance filtering, dedupe, and a FinBERT *classifier* (positive/negative/neutral only, so it cannot
   invent text), or, when its weights are not on disk (the corporate proxy blocks Hugging Face's weight
   CDN), the Loughran-McDonald finance dictionary (`FINBERT_PATH=/local/dir` enables FinBERT; the app
   never downloads at runtime). Needs 5+ headlines from 3+ sources or it says "insufficient".
   News can't be backtested live, so it only vetoes (score < -0.30), never proves. A separate **historical
   study** (`news_history.py`, `sentiment_study.py`) harvests ~3 years of monthly headlines per stock and
   tests sentiment and event flags against returns: no lookahead (only news from earlier days), a
   past-return placebo, non-overlapping t-stats, Bonferroni correction.
3. **ML filter** (`meta.py`): predicts whether the rule's own simulated trade wins. Features are pruned
   inside each training window (same sign across 3 time slices, top 8). Chosen on the selection period,
   then judged once on an untouched hold-out and again on stocks it never trained on. Adopted only if the
   filtered trades show a significant edge in both (`verdict()`).

`verdict.py` combines these: a rule-based Buy that fails validation is shown as **Watchlist** (with the
failed checks listed as "waiting for"), and appears under "Rule-based candidates (not validated)".

**Current findings (19 Oct 2026):**

* Backtest: 2,929 Buy trades over 9 years, edge vs random +0.04% (t = 0.33), positive in 50% of years.
  Last 3 years: +0.32% (t = 1.55), still below the 1.65 bar; on the final hold-out the rule's edge was
  negative (-0.36%). The edge fades on newer data.
* ML filter: it did not improve the rule on the hold-out (t = -1.17) or on unseen stocks (t = -0.57).
  The older excess-return model (`ml.py`, kept as a tested baseline) was also at chance (AUC 0.50).
* News: no sentiment or event feature predicts 5/10/20-day returns (nothing near the corrected
  |t| >= 3.26). 30-day sentiment correlates strongly with the *past* 20-day return (t = +5.4): news
  reports moves that already happened. Caveat: historical coverage is thin (25% of stock-weeks in 2023,
  74% in 2026) and the scorer is a dictionary, so this is absence of evidence, not proof of absence.

So nothing is currently validated as a Buy. That is the system working, not failing.

`python -m scripts.backtest_sanity` checks the engine can detect a planted edge (an oracle) and prints
per-feature information coefficients.

## Tests

```bash
.venv/bin/python -m pytest          # 156 tests, no network needed (~1 min; ML tests are slowest)
node tests/js/position-size.test.js
```

## Known limits

Not backtested. Large caps only. No news/earnings-date awareness. Yahoo data can lag or be wrong
(e.g. corporate actions such as the Tata Motors demerger).

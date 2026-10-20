# How to actually trade off this app

This is a walk-through for *using* the signals, not a repeat of the architecture docs in the
main [README](../README.md). Read that first if you want to know how a score is computed; read
this if you want to know what to click, when, and why.

**Not investment advice.** This tool surfaces evidence and gets out of the way; the decision,
the capital, and the risk are yours.

---

## 0. The one rule that matters more than any indicator

> A rule-based "Buy" or "setup" is only ever a **starting point**. The app downgrades it
> automatically the moment the evidence disagrees (see "Reading a downgrade" below). **Never
> override that downgrade with your own gut feel** — the whole point of the evidence layer is
> that gut feel is what loses money.

If you remember nothing else: **Strong Buy / Buy / Long setup / Short setup = validated by
rules AND backtest. Everything else is a "maybe, with reasons" — treat it as research, not a
trade ticket.**

---

## 1. Swing / positional trades (holding days to months)

### Step 1 — Open the dashboard (`/`)
Check **Market regime** first (top of the page). A "Correction / Bearish" regime is a message to
size down and be selective, not a reason to stop reading — the gates below already account for
price action, but they don't know your personal risk budget.

### Step 2 — Work the **Buy candidates** list top to bottom
Each card shows:
- **Composite score** (0-100) and the three pillar scores (Fundamental / Technical / Chart).
- The **verdict badge**. Only `Strong Buy` and `Buy` are suggestions to act on; see §3.

### Step 3 — Open the stock page, read the gates before the chart
Click through to `/stock/{symbol}`. Scroll to the **gates** section before you look at anything
else — it lists exactly which thresholds passed and which didn't, e.g.:

| Gate | Needs |
|---|---|
| Composite / Fundamental / Technical / Chart | >= 57 / 52 / 52 / 45 (Buy), higher for Strong Buy |
| Reward/risk | >= 1.5 (2.0 for Strong Buy) |
| No hard red flags | Falling knife, Loss-making, High leverage, Weak Piotroski all veto |
| Not extended (Strong Buy only) | < 3 ATR above the 20-EMA — don't chase a spike |

If `report.verdict` != `report.rule_verdict`, the rules fired but validation overruled them —
read §4 before doing anything.

### Step 4 — Take the trade plan as given, not as a guideline to improve on
The stock page prints **Entry / Stop / Target 1 / Target 2 / Reward:Risk**. The stop is placed
at a real support level or an ATR multiple — moving it further away "to give it room" is the
single most common way a planned 1.5R loss becomes a 4R loss.

### Step 5 — Size the position, place the order, walk away
Use the **position-size calculator** on the dashboard (same tool as intraday, see §2.5): capital
× risk% ÷ (entry − stop) = shares. Place entry + stop together. Checking the stock every hour
doesn't improve the outcome; it just tempts you to override the stop.

### Step 6 — Exits
- **Target 1**: scale out (common practice: half the position) and move the stop to breakeven.
- **Target 2**: let the rest run or take profit, your call going in, not mid-trade.
- **Stop hit**: it's a loss, not a signal to "wait and see" — exit and move on.
- Re-check the stock page occasionally: if the verdict degrades to Watchlist/Hold/Avoid on fresh
  data, that's informational (your stop still governs the exit), not an automatic sell signal.

---

## 2. Intraday trades (same-session, 5-minute bars)

### Step 1 — Check the market status banner at the top of `/intraday`
It tells you the phase in plain words: *Pre-market*, *Opening range forming* (wait, don't enter),
*Open: trading window* (the only phase accepting new entries), *Open: no new entries* (14:45
onward — manage existing positions only), *Closed*.

### Step 2 — Check **Active setups** before the full scanner table
This section only shows setups that are:
1. A live playbook trigger (see the 7 playbooks under "Intraday playbook" further down the page), **and**
2. Backtest-supported for that exact playbook + side (the **evidence badge** under the trigger text), **and**
3. Reward/risk >= 1.5, **and** aligned with the day's bias.

These are the closest thing this app has to "go trade this". Everything else in the Scanner
table below is either waiting on a rule, waiting on validation, or explicitly not tradeable.

### Step 3 — Reading the Scanner table's new confidence columns
Added so you don't have to click into every stock to sanity-check a verdict:

| Column | What it tells you | How to use it |
|---|---|---|
| **Backtest** | This exact playbook+side, replayed historically across **all 63 stocks pooled together** (not just this one) — trade count, average return after costs, day-clustered t-stat | Pooled because any single stock only fires a setup a handful of times over the archive, not enough to trust statistically; pooling gives real sample size at the cost of being a statement about the *strategy*, not this specific stock. Every stock currently showing the same playbook+side will show this **identical** number. "supported" = the historical edge clears the bar. "not supported" = this setup has **lost money** historically (a real, significant negative, not just "inconclusive") even though it's firing right now — that's why the verdict says "candidate" not "setup". "no live setup" just means nothing is firing, nothing to judge. |
| **Sentiment** | Daily news headline sentiment (Loughran-McDonald dictionary), same number the swing page uses | This is a **daily**, not intraday, signal — a stock can have very positive news and still chop sideways for an hour. Use it as a tie-breaker between two similar setups, never as an entry trigger by itself. Hover for headline count/sources: under 5 headlines / 3 sources means "thin coverage", treat as no signal. |
| **ML** | The project's machine-learning trade-outcome filter | Will read **"not used"** for the foreseeable future — the model has been tested against a strict hold-out and unseen-stock validation and failed both (see `/research` for the numbers). This is the honest state, not a bug. If it's ever adopted, the column will start showing a real per-stock win probability instead. |

**Bottom line on the three columns:** Backtest is the one that should change your behaviour
today. Sentiment is a secondary filter. ML is currently just documentation of "we tried, it
didn't work" — don't wait for a number that isn't coming.

### Step 4 — If a "candidate" (not a "setup") still tempts you
A `Long candidate` / `Short candidate` verdict means the live rules fired but the Backtest
column says "not supported" or "not backtested". Historically, trading this exact setup+side has
either lost money or has never been tested. If you take it anyway:
- Size smaller than you would for a validated "setup" (this is uncompensated risk, not a free edge).
- Don't average down — there's no statistical backstop telling you it mean-reverts.

### Step 5 — Position size and place the order
Use the **sizer** form at the top of the Active setups section (capital, risk% per trade). It
computes shares from the entry/stop on each card, capped by what your capital can actually buy
(no leverage assumed). Risk at most 1% of capital per trade.

### Step 6 — Non-negotiable intraday discipline (also printed on the page itself)
- Set a daily loss limit (e.g. 3R) and **stop trading for the day** when it's hit.
- Place the stop with the order. Never widen it. Never average down a loser.
- No new entries after 14:45 IST. Square off everything before the broker's auto square-off
  (roughly 15:15-15:20).
- Yahoo's 5-minute bars can lag by a few minutes — confirm the actual price in your broker
  terminal before you click buy/sell.

---

## 3. Verdict cheat-sheet (both pages)

| Verdict | Meaning | Action |
|---|---|---|
| **Strong Buy** / **Long setup** / **Short setup** | Rules fired *and* backtest/news/ML all agree | The closest thing to a go-ahead. Still size per your own risk rules. |
| **Buy** | Same as above, one tier down on conviction | Go-ahead, standard position size. |
| **Long candidate** / **Short candidate** | Live rules fired, but backtest/news/ML disagreed | Research-grade, not trade-grade. Read §2.4 (intraday) or §4 (swing) before touching it. |
| **Watchlist** | Good business, chart/technical not ready, OR a Buy was downgraded by validation | Wait. The stock page lists exactly what it's "waiting for". |
| **Hold** | Composite mediocre, no hard red flag | No action either way. |
| **Avoid** / **No trade** | Hard red flag, or no setup firing, or market phase doesn't allow entries | Don't force a trade; the "Why" column always says the specific reason. |

## 4. Reading a downgrade (swing page)

When `rule_verdict` says Buy but `verdict` says Watchlist, the stock page's gate list will show
one or more of these, each with the actual number next to it:
- **Backtest (universe or this stock)** — the price rule hasn't beaten random entries at a
  95% one-sided confidence level, or this stock's own trade history doesn't back it up.
- **News sentiment** — headline sentiment is below -0.30 (a real, not noisy, warning level).
- **ML model** — only appears when a model is adopted (currently: none).

This is the system doing exactly what it's for. A missing backtest is treated as *unvalidated*,
never as a free pass.

## 5. Where to dig deeper

- `/research` — the full evidence behind every gate: backtest methodology and results, the news
  sentiment correlation study, and the ML model's walk-forward numbers, including why it wasn't
  adopted.
- [README.md](../README.md) — architecture, scoring formulas, how each check is computed.
- `python -m scripts.backtest_sanity` — proves the backtest engine can detect a planted edge,
  if you want to sanity-check the research pipeline itself before trusting its verdicts.

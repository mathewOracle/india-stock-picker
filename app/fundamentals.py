"""Fundamental-analysis pillar, computed from 4-5 years of annual statements.

Profitability (ROE, margins), growth (3y CAGRs), balance-sheet health (debt,
interest cover), earnings quality (cash conversion, FCF), Piotroski F-score,
and valuation *relative to sector peers* (P/E, PEG) plus analyst consensus.
Banks/NBFCs skip the checks that don't apply to lenders.
"""

import math

from app.signals import Signal, grade

NAN = float("nan")
LOSS_PE = 1_000.0  # sentinel so loss-makers sort as "most expensive"


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NAN
    return number if math.isfinite(number) else NAN


def _div(a: float, b: float) -> float:
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else NAN


def _ok(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _row(statements: dict, section: str, *names: str) -> list[float]:
    rows = statements.get(section, {}).get("rows", {})
    for name in names:
        if name in rows:
            return [_num(v) for v in rows[name]]
    return []


def _at(values: list[float], i: int) -> float:
    return values[i] if i < len(values) else NAN


def _oldest(values: list[float], years: int) -> int | None:
    """Index of the oldest finite value within `years` of the newest (never 0)."""
    usable = [i for i in range(1, min(years, len(values) - 1) + 1) if math.isfinite(values[i])]
    return usable[-1] if usable else None


def cagr(values: list[float], years: int = 3) -> float:
    """Compound annual growth from the oldest usable year (<= `years` back) to values[0]."""
    n = _oldest(values, years)
    if n is None or not _ok(values[0]) or values[0] <= 0 or values[n] <= 0:
        return NAN
    return (values[0] / values[n]) ** (1 / n) - 1


def piotroski(s: dict) -> tuple[int, int]:
    """Piotroski F-score: (tests passed, tests we had data for). Max 9."""
    ni, rev = _row(s, "income", "Net Income Common Stockholders", "Net Income"), _row(s, "income", "Total Revenue")
    gross = _row(s, "income", "Gross Profit")
    assets, ltd = _row(s, "balance", "Total Assets"), _row(s, "balance", "Long Term Debt")
    ca, cl = _row(s, "balance", "Current Assets"), _row(s, "balance", "Current Liabilities")
    shares, ocf = _row(s, "balance", "Ordinary Shares Number"), _row(s, "cashflow", "Operating Cash Flow")

    def ratio(a, b, i):
        return _div(_at(a, i), _at(b, i))

    tests = [  # (left, right, strict): passes when left > right (or >= if not strict)
        (_at(ni, 0), 0.0, True),                              # profitable
        (_at(ocf, 0), 0.0, True),                             # cash-generative
        (ratio(ni, assets, 0), ratio(ni, assets, 1), True),   # ROA improving
        (_at(ocf, 0), _at(ni, 0), True),                      # cash > accounting profit
        (ratio(ltd, assets, 1), ratio(ltd, assets, 0), False),  # leverage not rising
        (ratio(ca, cl, 0), ratio(ca, cl, 1), True),           # liquidity improving
        (_at(shares, 1), _at(shares, 0), False),              # no dilution
        (ratio(gross, rev, 0), ratio(gross, rev, 1), True),   # gross margin improving
        (ratio(rev, assets, 0), ratio(rev, assets, 1), True),  # asset turnover improving
    ]
    usable = [(a, b, strict) for a, b, strict in tests if _ok(a, b)]
    passed = sum((a > b) if strict else (a >= b) for a, b, strict in usable)
    return passed, len(usable)


def fundamental_metrics(f: dict) -> dict:
    info, s = f.get("info", {}), f.get("statements", {})
    financial = info.get("sector") == "Financial Services"
    rev, ni = _row(s, "income", "Total Revenue"), _row(s, "income", "Net Income Common Stockholders", "Net Income")
    op, ebit = _row(s, "income", "Operating Income"), _row(s, "income", "EBIT")
    interest, eps = _row(s, "income", "Interest Expense"), _row(s, "income", "Diluted EPS")
    equity = _row(s, "balance", "Common Stock Equity", "Stockholders Equity")
    debt = _row(s, "balance", "Total Debt")
    ocf, fcf = _row(s, "cashflow", "Operating Cash Flow"), _row(s, "cashflow", "Free Cash Flow")

    margin_series = [_div(_at(op if not financial else ni, i), _at(rev, i)) for i in range(4)]
    pe = _num(info.get("trailingPE"))
    if not _ok(pe) and _ok(_at(ni, 0)):
        pe = _div(_num(info.get("marketCap")), _at(ni, 0))
    if _ok(pe) and pe <= 0:
        pe = LOSS_PE
    eps_growth = cagr(eps) if eps else cagr(ni)
    oldest_margin = _oldest(margin_series, 3)
    analysts = _num(info.get("numberOfAnalystOpinions"))
    piotroski_score, piotroski_max = (0, 0) if financial else piotroski(s)  # designed for non-lenders
    avg_equity = _at(equity, 0) if not _ok(_at(equity, 1)) else (_at(equity, 0) + _at(equity, 1)) / 2
    return {
        "sector": info.get("sector") or "Other",
        "industry": info.get("industry") or "",
        "financial": financial,
        "market_cap": _num(info.get("marketCap")),
        "roe": _div(_at(ni, 0), avg_equity),
        "net_margin": _div(_at(ni, 0), _at(rev, 0)),
        "margin_now": margin_series[0],
        "margin_change": margin_series[0] - margin_series[oldest_margin] if oldest_margin else NAN,
        "rev_cagr": cagr(rev),
        "eps_cagr": eps_growth,
        "rev_growth_1y": _div(_at(rev, 0), _at(rev, 1)) - 1,
        "de": NAN if financial else _div(_at(debt, 0), _at(equity, 0)),
        "interest_cover": NAN if financial else _div(_at(ebit, 0), _at(interest, 0)),
        "cash_conversion": NAN if financial or not _at(ni, 0) > 0 else _div(_at(ocf, 0), _at(ni, 0)),
        "fcf_positive_years": None if financial or not fcf else sum(v > 0 for v in fcf[:4] if _ok(v)),
        "fcf_yield": NAN if financial else _div(_at(fcf, 0), _num(info.get("marketCap"))),
        "piotroski": piotroski_score, "piotroski_max": piotroski_max,
        "pe": pe,
        "forward_pe": _num(info.get("forwardPE")),
        "pb": _num(info.get("priceToBook")),
        "ev_ebitda": _num(info.get("enterpriseToEbitda")),
        "peg": _div(pe, eps_growth * 100) if _ok(pe) and pe < LOSS_PE and eps_growth > 0 else NAN,
        "div_yield": _num(info.get("dividendYield")),
        "analyst_rec": _num(info.get("recommendationMean")),
        "analyst_key": info.get("recommendationKey") or "",
        "analysts": int(analysts) if _ok(analysts) else 0,
        "target_price": _num(info.get("targetMeanPrice")),
        "earnings_ts": _num(info.get("earningsTimestampStart")),  # next results date (unix seconds)
    }


# ---------- checks ----------

def _growth_signal(name: str, value: float, weight: float) -> Signal | None:
    if not _ok(value):
        return None
    bands = [(0.15, 1, "Compounding fast."), (0.10, 0.5, "Healthy growth."),
             (0.05, 0, "Moderate growth."), (0, -0.5, "Sluggish growth.")]
    pts, note = next(((p, n) for t, p, n in bands if value >= t), (-1, "Shrinking."))
    return Signal(name, f"{value:+.1%} a year", pts, weight, note)


def analyze_fundamentals(m: dict, sector_pe: float, price: float) -> list[Signal]:
    signals: list[Signal | None] = []
    add = signals.append

    if _ok(m["roe"]):
        add(Signal("Return on equity", f"{m['roe']:.1%}",
                   grade(m["roe"], [(0.20, 1), (0.15, 0.5), (0.10, 0)], below=-0.7 if m["roe"] > 0 else -1),
                   2.0, "Excellent capital efficiency." if m["roe"] >= 0.2 else "ROE above 15% is the quality bar."))
    add(_growth_signal("Revenue growth (3y CAGR)", m["rev_cagr"], 1.5))
    add(_growth_signal("EPS growth (3y CAGR)", m["eps_cagr"], 1.5))
    if _ok(m["margin_change"]):
        change = m["margin_change"]
        add(Signal("Margin trend (3y)", f"{m['margin_now']:.1%} now, {change * 100:+.1f} pp",
                   1.0 if change > 0.01 else -1.0 if change < -0.01 else 0.0, 1.0,
                   "Margins expanding: pricing power." if change > 0.01
                   else "Margins shrinking: cost or competition pressure." if change < -0.01 else "Stable margins."))
    if _ok(m["de"]):
        add(Signal("Debt / equity", f"{m['de']:.2f}x", grade(-m["de"], [(-0.3, 1), (-1, 0.3), (-2, -0.5)], below=-1),
                   1.5, "Near debt-free." if m["de"] < 0.3 else "Moderate debt." if m["de"] < 1
                   else "Leverage is a risk if rates or demand turn."))
    if _ok(m["interest_cover"]) and m["interest_cover"] > 0:
        add(Signal("Interest coverage", f"{m['interest_cover']:.1f}x",
                   grade(m["interest_cover"], [(8, 1), (3, 0.3), (1.5, -0.5)], below=-1), 1.0,
                   "Interest is easily serviced." if m["interest_cover"] >= 8 else "Thin cushion over interest costs."))
    if _ok(m["cash_conversion"]):
        add(Signal("Cash conversion (OCF / profit)", f"{m['cash_conversion']:.2f}x",
                   grade(m["cash_conversion"], [(0.9, 1), (0.6, 0)], below=-1), 1.0,
                   "Profits are backed by real cash." if m["cash_conversion"] >= 0.9 else "Profits not turning into cash."))
    if m["fcf_positive_years"] is not None:
        years = m["fcf_positive_years"]
        add(Signal("Free cash flow", f"positive {years} of last 4 yrs",
                   grade(years, [(3, 1), (2, 0)], below=-0.7), 1.0,
                   "Consistently self-funding." if years >= 3 else "Burns cash in many years."))
    if m["piotroski_max"] >= 5:
        ratio = m["piotroski"] / m["piotroski_max"]
        add(Signal("Piotroski F-score", f"{m['piotroski']}/{m['piotroski_max']}",
                   grade(ratio, [(0.75, 1), (0.55, 0.3), (0.35, -0.5)], below=-1), 2.0,
                   "Financials improving across the board." if ratio >= 0.75
                   else "Mixed financial trends." if ratio >= 0.55 else "Deteriorating financial trends."))
    if _ok(m["pe"]):
        if m["pe"] >= LOSS_PE:
            add(Signal("P/E vs sector", "loss-making", -1.0, 2.0, "No earnings to value."))
        elif _ok(sector_pe):
            rel = m["pe"] / sector_pe
            add(Signal("P/E vs sector", f"{m['pe']:.1f} vs sector {sector_pe:.1f}",
                       grade(-rel, [(-0.8, 1), (-1.1, 0.3), (-1.5, -0.3)], below=-1), 2.0,
                       "Cheaper than peers." if rel < 0.8 else "Pricier than peers." if rel > 1.1 else "In line with peers."))
    if _ok(m["peg"]):
        add(Signal("PEG ratio", f"{m['peg']:.2f}", grade(-m["peg"], [(-1, 1), (-2, 0.3), (-3, -0.3)], below=-1), 1.0,
                   "Growth is cheap." if m["peg"] < 1 else "Paying up for growth." if m["peg"] > 2 else "Fair price for growth."))
    if _ok(m["forward_pe"]) and _ok(m["pe"]) and m["pe"] < LOSS_PE:
        expanding = m["forward_pe"] < m["pe"]
        add(Signal("Forward vs trailing P/E", f"{m['forward_pe']:.1f} fwd vs {m['pe']:.1f}",
                   0.8 if expanding else -0.5, 0.5,
                   "Analysts expect earnings to grow." if expanding else "Analysts expect earnings to fall."))
    if m["analysts"] >= 3 and _ok(m["analyst_rec"], m["target_price"]):
        upside = m["target_price"] / price - 1
        pts = 1.0 if m["analyst_rec"] <= 2 and upside > 0.10 else 0.5 if m["analyst_rec"] <= 2.5 else -0.7 if m["analyst_rec"] > 3 else 0.0
        if upside < 0:
            pts = min(pts, -0.5)
        add(Signal("Analyst consensus", f"{m['analyst_key'] or m['analyst_rec']} ({m['analysts']}), target {upside:+.0%}",
                   pts, 1.0, "Street is bullish with upside." if pts >= 0.5 else "Street is lukewarm."))
    return [s for s in signals if s is not None]

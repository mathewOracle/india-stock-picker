"""FastAPI app: swing dashboard, stock deep-dive, intraday scanner, and background refresh jobs."""

import math
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import analysis, data
from app.intraday import analysis as intraday
from app.intraday import archive as intraday_archive
from app.intraday.checks import MIN_ATR_PCT, MIN_TRADED_VALUE_CR
from app.intraday.setups import PLAYBOOKS
from app.jobs import RefreshJob
from app.research import pipeline
from app.research.ml import PROBA_GATE
from app.scoring import GATES, PILLAR_WEIGHTS, VERDICTS, WATCHLIST
from app.universe import STOCKS

BASE = Path(__file__).resolve().parent
app = FastAPI(title="India Stock Picker")
app.add_middleware(GZipMiddleware, minimum_size=1024)  # stock pages carry ~140 KB of chart JSON
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
jobs = {"daily": RefreshJob(data.refresh), "intraday": RefreshJob(intraday_archive.refresh),
        "research": RefreshJob(pipeline.run_all)}

BUY_VERDICTS = ("Strong Buy", "Buy")


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value, signed: bool = True, digits: int = 1) -> str:
    return "–" if _is_nan(value) else f"{value * 100:{'+' if signed else ''}.{digits}f}%"


def num(value, digits: int = 1) -> str:
    return "–" if _is_nan(value) else f"{value:,.{digits}f}"


def pe(value) -> str:
    return "loss" if not _is_nan(value) and value >= 1000 else num(value)


def crore(value) -> str:
    return "–" if _is_nan(value) else f"₹{value / 1e7:,.0f} Cr"


def timestamp(value) -> str:
    return "never" if value is None else datetime.fromtimestamp(value).strftime("%d %b %Y, %H:%M")


templates.env.filters.update(pct=pct, num=num, pe=pe, crore=crore, timestamp=timestamp)


# ---------- swing / positional ----------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    result = analysis.build()
    ctx = {
        "a": result, "nav": "swing",
        "buys": [r for r in result.reports if r.verdict in BUY_VERDICTS],
        "candidates": [r for r in result.reports if r.rule_verdict in BUY_VERDICTS and r.verdict not in BUY_VERDICTS],
        "watchlist": [r for r in result.reports if r.verdict == "Watchlist"],
        "reports": result.reports,
        "sectors": sorted({r.metrics["sector"] for r in result.reports}),
        "verdicts": VERDICTS,
        "gates": GATES, "weights": PILLAR_WEIGHTS, "watch_rule": WATCHLIST,
    }
    return templates.TemplateResponse(request, "dashboard.html", ctx)


@app.get("/table", response_class=HTMLResponse)
def table(request: Request, sector: str = "", verdict: str = ""):
    reports = [r for r in analysis.build().reports
               if (not sector or r.metrics["sector"] == sector) and (not verdict or r.verdict == verdict)]
    return templates.TemplateResponse(request, "_table.html", {"reports": reports})


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock(request: Request, symbol: str):
    report = analysis.build().get(symbol)
    if report is None:
        raise HTTPException(404, "Unknown symbol, or no data yet - hit Refresh on the dashboard")
    ctx = {"r": report, "chart": analysis.chart_payload(symbol, report), "weights": PILLAR_WEIGHTS,
           "nav": "swing", "has_intraday": intraday.build().get(symbol) is not None}
    return templates.TemplateResponse(request, "stock.html", ctx)


# ---------- intraday ----------

@app.get("/intraday", response_class=HTMLResponse)
def intraday_dashboard(request: Request):
    scan = intraday.build()
    ctx = {"scan": scan, "nav": "intraday", "playbooks": PLAYBOOKS.values(),
           "setups": [r for r in scan.reports if r.verdict.endswith("setup")],
           "min_rr": intraday.MIN_RR, "min_value": MIN_TRADED_VALUE_CR, "min_atr": MIN_ATR_PCT,
           "long_bias": intraday.LONG_BIAS, "short_bias": intraday.SHORT_BIAS}
    return templates.TemplateResponse(request, "intraday.html", ctx)


@app.get("/intraday/{symbol}", response_class=HTMLResponse)
def intraday_stock(request: Request, symbol: str):
    scan = intraday.build()
    report = scan.get(symbol)
    if report is None:
        raise HTTPException(404, "Unknown symbol, or no intraday data yet - refresh on the Intraday page")
    ctx = {"r": report, "status": scan.status, "chart": intraday.chart_payload(report), "nav": "intraday"}
    return templates.TemplateResponse(request, "intraday_stock.html", ctx)


# ---------- research & validation ----------

@app.get("/research", response_class=HTMLResponse)
def research_page(request: Request):
    news = data.load_result("news") or {}
    scored = sorted(((s, v) for s, v in news.get("stocks", {}).items() if v["status"] == "ok"),
                    key=lambda item: item[1]["score"])
    ctx = {"nav": "research", "bt": data.load_result("backtest"), "news": news, "ml": data.load_result("model"),
           "study": data.load_result("sentiment_study"), "ib": data.load_result("intraday_backtest"),
           "ml_note": analysis.build().ml_note, "names": STOCKS, "proba_gate": PROBA_GATE,
           "news_low": scored[:5], "news_high": scored[::-1][:5]}
    return templates.TemplateResponse(request, "research.html", ctx)


# ---------- background refresh jobs ----------

def _job(name: str) -> RefreshJob:
    if name not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[name]


def _status(request: Request, name: str) -> HTMLResponse:
    status = _job(name).snapshot()
    response = templates.TemplateResponse(request, "_refresh_status.html", {"s": status, "name": name})
    if status.state == "done":
        response.headers["HX-Refresh"] = "true"  # new data -> reload the page
    return response


@app.post("/refresh/{name}", response_class=HTMLResponse)
def refresh(request: Request, name: str):
    _job(name).start()  # no-op if already running; either way we report live status
    return _status(request, name)


@app.get("/refresh/{name}/status", response_class=HTMLResponse)
def refresh_status(request: Request, name: str):
    return _status(request, name)

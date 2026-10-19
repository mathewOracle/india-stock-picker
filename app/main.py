"""FastAPI app: dashboard, stock deep-dive, and a background refresh with live status."""

import math
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import analysis, data
from app.jobs import RefreshJob
from app.scoring import GATES, PILLAR_WEIGHTS, VERDICTS, WATCHLIST

BASE = Path(__file__).resolve().parent
app = FastAPI(title="India Stock Picker")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
refresh_job = RefreshJob(data.refresh)

BUY_VERDICTS = ("Strong Buy", "Buy")


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value, signed: bool = True) -> str:
    return "–" if _is_nan(value) else f"{value * 100:{'+' if signed else ''}.1f}%"


def num(value, digits: int = 1) -> str:
    return "–" if _is_nan(value) else f"{value:,.{digits}f}"


def pe(value) -> str:
    return "loss" if not _is_nan(value) and value >= 1000 else num(value)


def crore(value) -> str:
    return "–" if _is_nan(value) else f"₹{value / 1e7:,.0f} Cr"


def timestamp(value) -> str:
    return "never" if value is None else datetime.fromtimestamp(value).strftime("%d %b %Y, %H:%M")


templates.env.filters.update(pct=pct, num=num, pe=pe, crore=crore, timestamp=timestamp)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    result = analysis.build()
    ctx = {
        "a": result,
        "buys": [r for r in result.reports if r.verdict in BUY_VERDICTS],
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
    ctx = {"r": report, "chart": analysis.chart_payload(symbol, report), "weights": PILLAR_WEIGHTS}
    return templates.TemplateResponse(request, "stock.html", ctx)


def _status(request: Request) -> HTMLResponse:
    status = refresh_job.snapshot()
    response = templates.TemplateResponse(request, "_refresh_status.html", {"s": status})
    if status.state == "done":
        response.headers["HX-Refresh"] = "true"  # new data -> reload the page
    return response


@app.post("/refresh", response_class=HTMLResponse)
def refresh(request: Request):
    refresh_job.start()  # no-op if already running; either way we report live status
    return _status(request)


@app.get("/refresh/status", response_class=HTMLResponse)
def refresh_status(request: Request):
    return _status(request)

"""FastAPI app: dashboard, stock detail, and a refresh button. Thin on purpose."""

import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from app import analysis, data
from app.scoring import FACTORS, WEIGHTS
from app.universe import STOCKS

BASE = Path(__file__).resolve().parent
app = FastAPI(title="India Stock Picker")
templates = Jinja2Templates(directory=BASE / "templates")

VERDICT_ORDER = ["Strong Buy", "Buy", "Hold", "Avoid"]


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


templates.env.filters.update(pct=pct, num=num, pe=pe, crore=crore)


def _filtered(picks: list[dict], sector: str, verdict: str) -> list[dict]:
    return [
        p for p in picks
        if (not sector or p["sector"] == sector) and (not verdict or p["verdict"] == verdict)
    ]


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    result = analysis.build()
    ctx = {
        "a": result,
        "top": [p for p in result.picks if p["verdict"] in ("Strong Buy", "Buy")][:8],
        "picks": result.picks,
        "sectors": sorted({p["sector"] for p in result.picks}),
        "verdicts": VERDICT_ORDER,
    }
    return templates.TemplateResponse(request, "dashboard.html", ctx)


@app.get("/table", response_class=HTMLResponse)
def table(request: Request, sector: str = "", verdict: str = ""):
    picks = _filtered(analysis.build().picks, sector, verdict)
    return templates.TemplateResponse(request, "_table.html", {"picks": picks})


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock(request: Request, symbol: str):
    if symbol not in STOCKS:
        raise HTTPException(404, "Unknown symbol")
    pick = next((p for p in analysis.build().picks if p["symbol"] == symbol), None)
    close = data.load_close(symbol)
    if pick is None or close is None:
        raise HTTPException(404, "No data yet - hit Refresh on the dashboard")
    sma200 = close.rolling(200).mean()
    chart = {
        "labels": [d.strftime("%Y-%m-%d") for d in close.index],
        "price": [round(v, 2) for v in close],
        "sma200": [None if _is_nan(v) else round(v, 2) for v in sma200],
    }
    ctx = {"p": pick, "chart": chart, "factors": FACTORS, "weights": WEIGHTS}
    return templates.TemplateResponse(request, "stock.html", ctx)


@app.post("/refresh")
def refresh():
    data.refresh()  # sync def -> runs in FastAPI's threadpool, doesn't block the loop
    return Response(headers={"HX-Refresh": "true"})

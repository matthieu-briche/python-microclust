"""
Données réelles gratuites : archives publiques Binance (https://data.binance.vision).
À lancer sur TA machine (l'environnement où ce code a été écrit n'avait pas accès au site).

aggTrades = transactions agrégées par ordre agressif, avec le côté de l'initiateur
(is_buyer_maker) -> c'est exactement ce qu'il faut pour le signe des ordres.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data"
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "best_match"]


def _url(symbol: str, day: date, market: str) -> str:
    root = "spot" if market == "spot" else "futures/um"
    return f"{BASE}/{root}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"


def load_day(symbol: str, day: date, market: str = "spot", cache: str | Path = "data") -> pd.DataFrame:
    cache = Path(cache) / market / symbol
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{day:%Y-%m-%d}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    with urllib.request.urlopen(_url(symbol, day, market), timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore")
    header = 0 if first[:1].isalpha() else None  # les archives futures ont un en-tête
    df = pd.read_csv(io.BytesIO(raw), header=header)
    df = df.iloc[:, :7]
    df.columns = COLS[:7]
    t = df["time"].astype("int64")
    unit = 1e6 if t.max() > 1e14 else 1e3  # spot : microsecondes depuis 2025, sinon ms
    out = pd.DataFrame({
        "time": (t - t.iloc[0]) / unit,                    # secondes
        "price": df["price"].astype(float),
        "qty": df["qty"].astype(float),
        "sign": np.where(df["is_buyer_maker"].astype(str).str.lower() == "true", -1, 1).astype(np.int8),
    })
    out.to_parquet(f)
    return out


def load_range(symbol: str, start: date, n_days: int, market: str = "spot", cache="data") -> pd.DataFrame:
    parts, offset = [], 0.0
    for i in range(n_days):
        d = load_day(symbol, start + timedelta(days=i), market, cache)
        d = d.assign(time=d["time"] + offset)
        offset = d["time"].iloc[-1] + 1.0
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def tick_sizes(symbols, market: str = "spot") -> dict:
    url = ("https://api.binance.com/api/v3/exchangeInfo" if market == "spot"
           else "https://fapi.binance.com/fapi/v1/exchangeInfo")
    with urllib.request.urlopen(url, timeout=60) as r:
        info = json.load(r)
    out = {}
    for s in info["symbols"]:
        if s["symbol"] in symbols:
            for flt in s["filters"]:
                if flt["filterType"] == "PRICE_FILTER":
                    out[s["symbol"]] = float(flt["tickSize"])
    return out

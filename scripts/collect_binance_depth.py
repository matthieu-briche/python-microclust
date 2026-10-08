"""
Collecteur L2 + trades en temps réel (Binance spot), pour les features de carnet
(taux d'annulation, profondeur au meilleur prix...). À faire tourner plusieurs
jours sur une machine allumée (un petit VPS suffit).

    pip install websockets pandas pyarrow
    python scripts/collect_binance_depth.py BTCUSDT ETHUSDT SOLUSDT --out data/l2

Écrit des fichiers parquet horaires : depth_<SYM>_<heure>.parquet (time, side,
price, qty = quantité ABSOLUE au niveau après mise à jour) et trades_<SYM>_<heure>.parquet.

NB : non testé contre le flux réel (pas d'accès réseau à Binance là où il a été
écrit). Points à vérifier au premier lancement : format des messages, reconnexion.
La resynchronisation stricte avec snapshot (lastUpdateId) n'est pas nécessaire ici :
on ne reconstruit pas le carnet, on n'a besoin que des variations par niveau,
et la première valeur vue à chaque niveau sert de référence.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd
import websockets

URL = "wss://stream.binance.com:9443/stream?streams="


async def run(symbols, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    streams = "/".join(f"{s.lower()}@depth@100ms/{s.lower()}@aggTrade" for s in symbols)
    buf = {s: {"depth": [], "trades": []} for s in symbols}
    hour = int(time.time() // 3600)

    def flush(h):
        for s, b in buf.items():
            for kind, rows in b.items():
                if rows:
                    pd.DataFrame(rows).to_parquet(out / f"{kind}_{s}_{h}.parquet")
                    rows.clear()

    while True:
        try:
            async with websockets.connect(URL + streams, ping_interval=20) as ws:
                async for msg in ws:
                    m = json.loads(msg)
                    d, stream = m["data"], m["stream"]
                    sym = d["s"]
                    if "@depth" in stream:
                        t = d["E"]
                        for side, key in (("b", "b"), ("a", "a")):
                            for px, q in d[key]:
                                buf[sym]["depth"].append((t, side, float(px), float(q)))
                    else:
                        buf[sym]["trades"].append(
                            (d["T"], float(d["p"]), float(d["q"]), -1 if d["m"] else 1))
                    h = int(time.time() // 3600)
                    if h != hour:
                        for s in buf:
                            buf[s]["depth"] = pd.DataFrame(buf[s]["depth"], columns=["time", "side", "price", "qty"]).to_dict("records")
                            buf[s]["trades"] = pd.DataFrame(buf[s]["trades"], columns=["time", "price", "qty", "sign"]).to_dict("records")
                        flush(hour)
                        hour = h
        except Exception as exc:  # reconnexion simple
            print("déconnecté :", exc, "- reconnexion dans 5 s")
            await asyncio.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--out", default="data/l2")
    a = ap.parse_args()
    asyncio.run(run([s.upper() for s in a.symbols], Path(a.out)))

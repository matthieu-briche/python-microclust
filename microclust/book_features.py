"""
Features "niveau 2" qui demandent le carnet d'ordres (pas seulement les trades).
Les données historiques gratuites n'existent quasiment pas : il faut les
COLLECTER soi-même (voir scripts/collect_binance_depth.py).

Taux d'annulation (Cartea, Jaimungal, Penalva, §4.4 "Messages and Cancellation
Activity") : avec un flux L2 on ne voit pas les ordres individuels, seulement la
quantité affichée à chaque niveau. On l'estime ainsi :
    baisse de quantité à un niveau  -  volume exécuté à ce prix  =  volume annulé
C'est une borne inférieure (un ajout et une annulation dans le même pas de 100 ms
se compensent), à documenter comme telle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cancel_features(depth: pd.DataFrame, trades: pd.DataFrame, bucket_ms: int = 100) -> pd.Series:
    """
    depth  : colonnes time(ms), side('b'/'a'), price, qty  -> quantité ABSOLUE après mise à jour
             (format du flux diff-depth Binance, appliqué sur un snapshot initial)
    trades : colonnes time(ms), price, qty, sign
    """
    d = depth.sort_values("time").copy()
    d["prev"] = d.groupby(["side", "price"])["qty"].shift(1)
    d = d.dropna(subset=["prev"])
    d["delta"] = d["qty"] - d["prev"]
    d["bucket"] = (d["time"] // bucket_ms).astype(np.int64)

    t = trades.copy()
    t["bucket"] = (t["time"] // bucket_ms).astype(np.int64)
    # un achat agressif consomme le côté ask, une vente le côté bid
    t["side"] = np.where(t["sign"] > 0, "a", "b")
    traded = t.groupby(["bucket", "side", "price"])["qty"].sum().rename("traded")

    removed = (-d.loc[d["delta"] < 0].groupby(["bucket", "side", "price"])["delta"].sum()).rename("removed")
    added = d.loc[d["delta"] > 0, "delta"].sum()
    m = pd.concat([removed, traded], axis=1).fillna(0.0)
    cancelled = (m["removed"] - m["traded"]).clip(lower=0).sum()
    vol = t["qty"].sum()
    return pd.Series({
        "cancel_to_trade": cancelled / vol if vol else np.nan,       # volume annulé / volume exécuté
        "cancel_to_add": cancelled / added if added else np.nan,     # part des ajouts finalement annulés
    })


def _selftest():
    # niveau ask 101 : 10 -> 4 (6 retirés, dont 2 exécutés => 4 annulés) ; ajout de 5 au bid 99
    depth = pd.DataFrame({
        "time": [0, 0, 150, 150],
        "side": ["a", "b", "a", "b"],
        "price": [101.0, 99.0, 101.0, 99.0],
        "qty": [10.0, 3.0, 4.0, 8.0],
    })
    trades = pd.DataFrame({"time": [120], "price": [101.0], "qty": [2.0], "sign": [1]})
    f = cancel_features(depth, trades)
    assert abs(f["cancel_to_trade"] - 2.0) < 1e-12, f
    assert abs(f["cancel_to_add"] - 0.8) < 1e-12, f
    return f


if __name__ == "__main__":
    print(_selftest())

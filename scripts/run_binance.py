"""
Première étude sur données RÉELLES (Binance, gratuites). À lancer chez toi :

    pip install -r requirements.txt
    python scripts/run_binance.py --start 2026-09-01 --days 3
    python scripts/run_binance.py --market futures --start 2026-09-01 --days 3

Le premier lancement télécharge ~quelques Go pour 30 paires x 3 jours (cache local ./data).
Conseil : commence avec --days 1 et --max-trades 300000 pour vérifier que tout tourne.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from microclust.binance import load_range, tick_sizes
from microclust.features import signature_table
from microclust.onc import onc, standardize, corr_to_obs, labels_from
from microclust.transfer import transfer_loss, euclid, mantel, split_stability

# Univers volontairement hétérogène : majors, grandes alts, "memecoins", stablecoin-ish
DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
           "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT",
           "ATOMUSDT", "UNIUSDT", "AAVEUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
           "APTUSDT", "INJUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "FDUSDUSDT", "ETCUSDT",
           "XLMUSDT", "HBARUSDT"]


def hourly_returns(trades: pd.DataFrame, t0: pd.Timestamp) -> pd.Series:
    t = t0 + pd.to_timedelta(trades["time"], unit="s")
    px = pd.Series(trades["price"].to_numpy(), index=t).resample("1h").last().ffill()
    return np.log(px).diff().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=DEFAULT)
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--market", choices=["spot", "futures"], default="spot")
    ap.add_argument("--max-trades", type=int, default=2_000_000,
                    help="tronque chaque série (le Hawkes est le plus lent)")
    a = ap.parse_args()
    start = date.fromisoformat(a.start)
    out = Path(__file__).resolve().parents[1] / "results" / f"binance_{a.market}_{a.start}_{a.days}d"
    out.mkdir(parents=True, exist_ok=True)

    try:
        ticks = tick_sizes(a.symbols, a.market)
    except Exception as exc:
        print("exchangeInfo indisponible, tick inféré depuis les prix :", exc)
        ticks = {}

    universe, rets = {}, {}
    for s in a.symbols:
        try:
            df = load_range(s, start, a.days, a.market)
        except Exception as exc:
            print(f"  {s} ignoré ({exc})")
            continue
        rets[s] = hourly_returns(df, pd.Timestamp(start))
        universe[s] = df.iloc[: a.max_trades].reset_index(drop=True)
        print(f"  {s}: {len(df):,} aggTrades")

    S = signature_table(universe, ticks)
    S.to_csv(out / "signatures.csv")
    Z = standardize(S)
    cl_m, _ = onc(Z)
    lab_m = labels_from(cl_m, Z.index)

    R = pd.DataFrame(rets).dropna()
    cl_r, _ = onc(corr_to_obs(R.corr()))
    lab_r = labels_from(cl_r, R.columns).reindex(Z.index)

    L = transfer_loss(universe)
    Lsym = (L + L.T) / 2
    r_m, p_m = mantel(Lsym, euclid(Z))
    r_r, p_r = mantel(Lsym, corr_to_obs(R.corr()).loc[Z.index, Z.index])
    stab = split_stability(universe, ticks)

    res = pd.DataFrame({"cluster_micro": lab_m, "cluster_rendements": lab_r}).join(S.round(4))
    res.sort_values("cluster_micro").to_csv(out / "clusters.csv")
    print(res.sort_values("cluster_micro")[["cluster_micro", "cluster_rendements"]])
    print(f"\nARI(micro, rendements)            = {ari(lab_m, lab_r):.3f}  (proche de 0 = partitions différentes)")
    print(f"Mantel perte~distance micro       : rho={r_m:.3f}  p={p_m:.4f}")
    print(f"Mantel perte~distance rendements  : rho={r_r:.3f}  p={p_r:.4f}")
    print(f"Stabilité 1re/2e moitié (ARI)     = {stab:.3f}")
    print(f"\nRésultats : {out}")


if __name__ == "__main__":
    main()

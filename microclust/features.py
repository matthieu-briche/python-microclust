"""
Signature de microstructure d'un actif = vecteur de grandeurs SANS DIMENSION,
calculé uniquement à partir des transactions (time, price, qty, sign).

Pourquoi sans dimension : pour comparer un BTC à 60 000 $ et un altcoin à 0,30 $,
chaque grandeur est normalisée par le spread, le tick ou une variance de
référence. Sans ça, le clustering regroupe par niveau de prix, pas par mécanique.

Références (Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices*) :
  gamma, c1         -> ch. 10 (mémoire longue du signe des ordres)
  impact1, persist  -> ch. 11 (fonction de réponse R(l)) et ch. 13 (propagateur)
  spread_ticks,
  p_move            -> §4.8 (effets de taille de tick, grand tick vs petit tick)
  sigma_spread      -> ch. 16 (relation volatilité/spread, "spread ~ vol par trade")
  vr50              -> ch. 2 (signature plot / ratio de variance)
  hawkes_n          -> ch. 9 (endogénéité, ratio de branchement)
Le taux d'annulation (Cartea et al., §4.4) nécessite le carnet : voir book_features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # repli lent mais fonctionnel
    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco

from scipy.optimize import minimize

FEATURES = ["gamma", "c1", "log_spread_ticks", "p_move", "impact1",
            "impact_persist", "sigma_spread", "vr50", "hawkes_n"]


# --------------------------------------------------------------------- helpers
def infer_tick(price: np.ndarray) -> float:
    u = np.unique(np.round(price, 10))
    d = np.diff(u)
    d = d[d > 0]
    return float(np.quantile(d, 0.01)) if d.size else np.nan


def acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return ac / ac[0]


def sign_memory_exponent(sign: np.ndarray) -> float:
    """gamma tel que C(l) ~ l^-gamma, estimé par la méthode de la variance agrégée :
    Var(somme des signes sur w trades) ~ w^(2H), gamma = 2 - 2H.
    Nettement moins bruité que l'ajustement direct de l'autocorrélation
    (testé sur signes simulés : écart-type ~2x plus faible à 30k trades)."""
    e = sign.astype(float) - sign.mean()
    ws = np.unique(np.logspace(1, np.log10(max(len(e) / 20, 20)), 15).astype(int))
    v = []
    for w in ws:
        n = len(e) // w
        v.append(e[: n * w].reshape(n, w).sum(1).var())
    H = np.polyfit(np.log(ws), np.log(v), 1)[0] / 2
    return float(2 - 2 * H)


def spread_proxy(price: np.ndarray, sign: np.ndarray) -> float:
    """Spread effectif estimé à partir des seuls trades : saut de prix médian
    entre deux transactions consécutives de signes opposés (rebond bid-ask)."""
    flip = sign[1:] != sign[:-1]
    jumps = np.abs(np.diff(price))[flip]
    jumps = jumps[jumps > 0]
    return float(np.median(jumps)) if jumps.size else np.nan


def response(mid: np.ndarray, sign: np.ndarray, lags) -> np.ndarray:
    """R(l) = E[eps_t (m_{t+l} - m_t)]  (Bouchaud ch. 11)."""
    out = []
    for l in lags:
        out.append(np.mean(sign[:-l] * (mid[l:] - mid[:-l])))
    return np.array(out)


# ---------------------------------------------------------------- Hawkes (MLE)
@njit(cache=True)
def _hawkes_nll(mu, a, b, t):
    # noyau a*exp(-b s) ; ratio de branchement n = a/b
    T = t[-1] - t[0]
    A = 0.0
    ll = np.log(mu)
    for i in range(1, t.size):
        A = np.exp(-b * (t[i] - t[i - 1])) * (1.0 + A)
        ll += np.log(mu + a * A)
    comp = mu * T + (a / b) * np.sum(1.0 - np.exp(-b * (t[-1] - t)))
    return -(ll - comp)


def hawkes_branching(times: np.ndarray, max_events: int = 50_000) -> float:
    t = np.asarray(times, float)
    t = t[: max_events]
    t = t - t[0]
    # instants identiques (horodatage à la ms) : on étale uniformément
    dup = np.diff(t, prepend=-1) <= 0
    if dup.any():
        t = t + np.random.default_rng(0).uniform(0, 1e-4, t.size)
        t = np.sort(t)
    rate = t.size / (t[-1] + 1e-12)

    def f(x):
        mu, n, b = np.exp(x[0]), 1 / (1 + np.exp(-x[1])), np.exp(x[2])
        return _hawkes_nll(mu, n * b, b, t)

    best = None
    for b0 in (rate * 0.1, rate, rate * 10):
        x0 = np.array([np.log(rate * 0.5), 0.0, np.log(b0)])
        r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return float(1 / (1 + np.exp(-best.x[1])))


# ------------------------------------------------------------------- signature
def signature(trades: pd.DataFrame, tick: float | None = None,
              hawkes: bool = True) -> pd.Series:
    p = trades["price"].to_numpy(float)
    e = trades["sign"].to_numpy(np.int8).astype(float)
    tick = tick if tick else infer_tick(p)
    s = spread_proxy(p, e)
    mid = p - e * s / 2.0  # proxy du mid : on retire le demi-spread

    ac = acf_fft(e, 1000)
    gamma = sign_memory_exponent(e)

    R = response(mid, e, [1, 100])
    dm = np.diff(mid)
    v1 = np.var(dm)
    v50 = np.var(mid[50:] - mid[:-50])
    v10 = np.var(mid[10:] - mid[:-10])

    out = {
        "gamma": gamma,
        "c1": ac[1],
        "log_spread_ticks": np.log(s / tick),
        "p_move": np.mean(np.abs(dm) > tick / 2),
        "impact1": R[0] / s,
        # persistance de l'impact R(100)/R(1), compressée dans (-1, 1) :
        # ~1 = impact permanent (carnet peu résilient), <=0 = impact transitoire
        "impact_persist": float(2 / np.pi * np.arctan(R[1] / (abs(R[0]) + 1e-12))),
        "sigma_spread": np.sqrt(v10 / 10) / s,
        "vr50": v50 / (50 * v1),
        "hawkes_n": hawkes_branching(trades["time"].to_numpy()) if hawkes else np.nan,
    }
    return pd.Series(out)


def signature_table(universe: dict, ticks: dict | None = None, hawkes=True) -> pd.DataFrame:
    rows = {k: signature(v, (ticks or {}).get(k), hawkes) for k, v in universe.items()}
    return pd.DataFrame(rows).T[FEATURES]

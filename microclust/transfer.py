"""
Le test qui donne du sens au projet : une stratégie calibrée sur l'actif A
se transfère-t-elle mieux vers B quand A et B sont proches en microstructure
qu'en corrélation de rendements ?

Stratégie témoin volontairement simple (et dépendante de la microstructure) :
  signal_t = EWMA des signes des ordres passés, demi-vie h (en transactions)
  cible_t  = m_{t+H} - m_t (variation du mid proxy sur H transactions)
  score    = IC = corr(signal, cible)
La demi-vie optimale h* dépend de la mémoire du signe (gamma) et de la forme du
propagateur : c'est exactement ce que la signature capture.

Perte de transfert : L(A->B) = IC_B(h*_B) - IC_B(h*_A)  (>= 0).
On teste ensuite, par un test de Mantel (permutation des actifs), si la matrice L
est corrélée à la distance de microstructure, et à la distance de corrélation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from .features import spread_proxy, signature_table
from .onc import onc, standardize, labels_from

HALF_LIVES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])


def ic_curve(trades: pd.DataFrame, horizon: int = 10, half_lives=HALF_LIVES) -> np.ndarray:
    p = trades["price"].to_numpy(float)
    e = trades["sign"].to_numpy(float)
    mid = p - e * spread_proxy(p, e) / 2
    fut = mid[horizon:] - mid[:-horizon]
    out = []
    for h in half_lives:
        lam = 0.5 ** (1 / h)
        sig = lfilter([1 - lam], [1, -lam], e)[:-horizon]
        out.append(np.corrcoef(sig, fut)[0, 1])
    return np.array(out)


def transfer_loss(universe: dict, horizon: int = 10) -> pd.DataFrame:
    curves = {k: ic_curve(v, horizon) for k, v in universe.items()}
    names = list(universe)
    best = {k: int(np.nanargmax(c)) for k, c in curves.items()}
    L = pd.DataFrame(0.0, index=names, columns=names)  # ligne = source A, colonne = cible B
    for a in names:
        for b in names:
            L.loc[a, b] = curves[b][best[b]] - curves[b][best[a]]
    return L


def euclid(X: pd.DataFrame) -> pd.DataFrame:
    v = X.values
    d = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(-1))
    return pd.DataFrame(d, index=X.index, columns=X.index)


def mantel(D1: pd.DataFrame, D2: pd.DataFrame, n_perm: int = 2000, seed: int = 0):
    """Corrélation de Spearman entre deux matrices de distance + p-valeur par permutation."""
    names = D1.index
    D2 = D2.loc[names, names]
    iu = np.triu_indices(len(names), 1)
    a = D1.values[iu]
    r0 = spearmanr(a, D2.values[iu]).statistic
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(names))
        r = spearmanr(a, D2.values[np.ix_(p, p)][iu]).statistic
        cnt += r >= r0
    return float(r0), (cnt + 1) / (n_perm + 1)


def split_stability(universe: dict, ticks: dict | None = None, seed: int = 0) -> float:
    """ARI entre les clusterings obtenus sur la 1re et la 2e moitié des données.
    Si les clusters de microstructure ne sont pas stables dans le temps, ils ne
    servent à rien pour transférer une stratégie."""
    halves = []
    for part in (0, 1):
        u = {k: (v.iloc[: len(v) // 2] if part == 0 else v.iloc[len(v) // 2:]).reset_index(drop=True)
             for k, v in universe.items()}
        S = signature_table(u, ticks, hawkes=True)
        cl, _ = onc(standardize(S), seed=seed)
        halves.append(labels_from(cl, S.index))
    return adjusted_rand_score(halves[0], halves[1])

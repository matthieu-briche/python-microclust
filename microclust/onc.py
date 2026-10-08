"""
ONC — Optimal Number of Clusters (López de Prado, *ML for Asset Managers*, ch. 4,
snippets 4.1 et 4.2), porté en Python 3 et généralisé.

Différence avec le livre : le livre part d'une matrice de corrélation et construit
la matrice d'observations X = sqrt(0.5 (1 - rho)). Ici `onc()` accepte directement
n'importe quelle matrice d'observations N x F (le livre l'autorise explicitement,
§4.4.2 : "or applying some other method"). On l'alimente :
  - soit avec les signatures de microstructure standardisées,
  - soit avec X dérivé d'une corrélation de rendements (`corr_to_obs`), pour comparer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def corr_to_obs(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1 - corr.fillna(0)) / 2.0) ** 0.5


def standardize(F: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """z-score robuste (médiane / MAD) puis écrêtage : les features de microstructure
    ont des queues épaisses entre actifs."""
    med = F.median()
    mad = (F - med).abs().median() * 1.4826
    Z = (F - med) / mad.replace(0, np.nan)
    return Z.clip(-clip, clip).fillna(0.0)


def _quality(silh: np.ndarray) -> float:
    sd = silh.std()
    return silh.mean() / sd if sd > 0 else -np.inf


def onc_base(X: pd.DataFrame, max_k: int = 10, n_init: int = 10, seed: int = 0):
    rng = np.random.RandomState(seed)
    best = (None, None, -np.inf)
    max_k = min(max_k, X.shape[0] - 1)
    for _ in range(n_init):
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=1, random_state=rng.randint(1 << 30)).fit(X.values)
            s = silhouette_samples(X.values, km.labels_)
            q = _quality(s)
            if q > best[2]:
                best = (km.labels_, s, q)
    labels, silh, _ = best
    clstrs = {i: X.index[labels == i].tolist() for i in np.unique(labels)}
    return clstrs, pd.Series(silh, index=X.index)


def _relabel_silh(X, clstrs):
    lab = np.zeros(len(X), int)
    for i, members in enumerate(clstrs.values()):
        lab[[X.index.get_loc(m) for m in members]] = i
    return pd.Series(silhouette_samples(X.values, lab), index=X.index)


def onc(X: pd.DataFrame, max_k: int | None = None, n_init: int = 10, seed: int = 0):
    """Clustering ONC récursif. Renvoie (dict cluster -> membres, silhouettes)."""
    if max_k is None:
        max_k = X.shape[0] - 1
    clstrs, silh = onc_base(X, max_k, n_init, seed)
    t = {i: silh[m].mean() / (silh[m].std() + 1e-12) for i, m in clstrs.items()}
    t_mean = np.mean(list(t.values()))
    redo = [i for i in t if t[i] < t_mean]
    if len(redo) <= 1:
        return clstrs, silh
    keys = [m for i in redo for m in clstrs[i]]
    if len(keys) < 3:
        return clstrs, silh
    sub, _ = onc(X.loc[keys], min(max_k, len(keys) - 1), n_init, seed + 1)
    new = [clstrs[i] for i in clstrs if i not in redo] + list(sub.values())
    new = {i: m for i, m in enumerate(new)}
    silh_new = _relabel_silh(X, new)
    t_new = np.mean([silh_new[m].mean() / (silh_new[m].std() + 1e-12) for m in new.values()])
    t_redo = np.mean([t[i] for i in redo])
    return (new, silh_new) if t_new > t_redo else (clstrs, silh)


def labels_from(clstrs: dict, index) -> pd.Series:
    lab = pd.Series(-1, index=index)
    for i, m in clstrs.items():
        lab[m] = i
    return lab

"""
Simulateur de flux de transactions synthétiques avec une microstructure CONTRÔLÉE.

But : valider la chaîne (features -> ONC) comme le fait López de Prado au §4.5 de
*ML for Asset Managers* : on injecte des familles connues, on vérifie que le
clustering les retrouve. Ce n'est PAS un modèle réaliste du marché.

Ingrédients (références : Bouchaud et al., *Trades, Quotes and Prices*) :
  - signes des ordres à mémoire longue via des "métaordres" de longueur Pareto
    (modèle de Lillo-Mike-Farmer, ch. 10) : C(l) ~ l^-gamma, gamma = alpha - 1 ;
  - prix "mid" latent = modèle du propagateur (ch. 13) :
        m_t = sum_{s<t} G(t-s) eps_s + bruit,  G(l) = G0 / (1+l)^beta ;
  - grille de prix discrète (tick) + spread naturel : grand tick vs petit tick (§4.8) ;
  - instants d'arrivée = processus de Hawkes exponentiel (ch. 9), ratio de branchement n.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


@dataclass
class MicroParams:
    alpha: float = 1.5        # queue Pareto des métaordres -> gamma = alpha - 1
    noise_frac: float = 0.3   # part des transactions à signe iid (traders "bruit")
    G0: float = 0.25          # impact instantané (en unités de spread naturel)
    beta: float = 0.25        # décroissance du propagateur
    tick: float = 0.05        # taille du tick (spread naturel = 1)
    hawkes_n: float = 0.6     # ratio de branchement (endogénéité)
    hawkes_decay: float = 1.0 # 1/temps caractéristique du noyau (s^-1)
    vol_noise: float = 0.05   # diffusion "news" par transaction


def _hawkes_times(mu: float, n: float, decay: float, T: float, rng) -> np.ndarray:
    """Simulation par branchement (représentation en grappes) d'un Hawkes exponentiel."""
    gen = rng.uniform(0, T, rng.poisson(mu * T))
    out = [gen]
    while gen.size:
        kids = rng.poisson(n, gen.size)
        parents = np.repeat(gen, kids)
        gen = parents + rng.exponential(1.0 / decay, parents.size)
        gen = gen[gen < T]
        out.append(gen)
    return np.sort(np.concatenate(out))


def _long_memory_signs(N: int, alpha: float, noise_frac: float, rng) -> np.ndarray:
    lens = []
    total = 0
    while total < N:
        L = np.ceil(rng.pareto(alpha, 4096) + 1).astype(int)
        lens.append(L)
        total += L.sum()
    L = np.concatenate(lens)
    signs = np.repeat(rng.choice([-1, 1], L.size), L)[:N]
    noise = rng.random(N) < noise_frac
    signs[noise] = rng.choice([-1, 1], noise.sum())
    return signs.astype(np.int8)


def simulate_asset(p: MicroParams, n_trades: int = 30_000, seed: int = 0) -> pd.DataFrame:
    """Renvoie un DataFrame (time, price, qty, sign) au même format que le loader Binance."""
    rng = np.random.default_rng(seed)
    mu = 1.0 * (1 - p.hawkes_n)  # intensité moyenne ~1 trade/s
    T = n_trades / 1.0 * 1.3
    t = _hawkes_times(mu, p.hawkes_n, p.hawkes_decay, T, rng)[:n_trades]
    N = t.size
    eps = _long_memory_signs(N, p.alpha, p.noise_frac, rng)

    lags = np.arange(N)
    G = p.G0 / (1.0 + lags) ** p.beta
    impact = fftconvolve(eps.astype(float), G)[:N]
    impact = np.concatenate([[0.0], impact[:-1]])  # seuls les ordres passés impactent
    m = 1000.0 + impact + np.cumsum(rng.normal(0, p.vol_noise, N))

    half = 0.5  # spread naturel = 1
    bid = np.floor((m - half) / p.tick) * p.tick
    ask = np.ceil((m + half) / p.tick) * p.tick
    ask = np.where(ask - bid < p.tick, bid + p.tick, ask)
    price = np.where(eps > 0, ask, bid)
    qty = rng.lognormal(0, 1, N)
    return pd.DataFrame({"time": t, "price": price, "qty": qty, "sign": eps})


# ---------------------------------------------------------------------------
# Familles de microstructure (ce que le clustering doit retrouver)
# ---------------------------------------------------------------------------
FAMILIES = {
    # grand tick, peu de mémoire, marché "calme"
    "A_large_tick": MicroParams(alpha=1.8, noise_frac=0.5, G0=0.15, beta=0.40, tick=2.0, hawkes_n=0.4),
    # petit tick, forte mémoire du signe (métaordres longs), impact persistant
    "B_small_tick_memory": MicroParams(alpha=1.3, noise_frac=0.2, G0=0.30, beta=0.15, tick=0.02, hawkes_n=0.6),
    # petit tick, impact transitoire fort (carnet résilient), très endogène
    "C_resilient_endogenous": MicroParams(alpha=1.7, noise_frac=0.4, G0=0.50, beta=0.70, tick=0.05, hawkes_n=0.85),
    # tick intermédiaire, mémoire moyenne
    "D_mid_tick": MicroParams(alpha=1.5, noise_frac=0.3, G0=0.25, beta=0.30, tick=0.5, hawkes_n=0.7),
}


def jitter(p: MicroParams, rng, scale: float = 0.06) -> MicroParams:
    """Petite dispersion intra-famille (aucun actif n'est la copie d'un autre)."""
    d = asdict(p)
    for k in ("alpha", "G0", "beta", "hawkes_n", "noise_frac"):
        d[k] *= float(np.exp(rng.normal(0, scale)))
    d["alpha"] = float(np.clip(d["alpha"], 1.1, 1.95))
    d["hawkes_n"] = float(np.clip(d["hawkes_n"], 0.05, 0.95))
    d["noise_frac"] = float(np.clip(d["noise_frac"], 0.0, 0.9))
    d["tick"] *= float(np.exp(rng.normal(0, 2 * scale)))
    return MicroParams(**d)


def make_universe(per_family: int = 8, n_trades: int = 30_000, seed: int = 42):
    """Univers synthétique : dict nom -> trades, et Series nom -> vraie famille."""
    rng = np.random.default_rng(seed)
    trades, truth = {}, {}
    i = 0
    for fam, base in FAMILIES.items():
        for j in range(per_family):
            name = f"{fam[0]}{j:02d}"
            trades[name] = simulate_asset(jitter(base, rng), n_trades, seed=seed * 1000 + i)
            truth[name] = fam
            i += 1
    return trades, pd.Series(truth, name="family")


def make_return_panel(names, n_days: int = 500, n_sectors: int = 3, seed: int = 7):
    """Rendements journaliers avec facteur marché + secteurs tirés INDÉPENDAMMENT
    des familles de microstructure. Sert à illustrer que les deux partitions
    peuvent différer (dans la réalité c'est justement la question empirique)."""
    rng = np.random.default_rng(seed)
    sector = pd.Series(rng.integers(0, n_sectors, len(names)), index=names)
    mkt = rng.normal(0, 0.01, n_days)
    sec = rng.normal(0, 0.01, (n_sectors, n_days))
    R = {n: mkt + sec[sector[n]] + rng.normal(0, 0.008, n_days) for n in names}
    return pd.DataFrame(R), sector

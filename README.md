# microclust: clustering assets by market mechanics, not by returns

> **Status: research prototype.** The measurement pipeline is validated on synthetic data and runs end to end on real Binance data. The main hypothesis (H3) has not yet been tested at scale on real markets.

## The idea

Two assets can be **uncorrelated in returns** yet **work the same way** at the microstructure level: same order-sign memory, same price impact, same tick regime, same endogeneity. If so, an intraday strategy calibrated on one should **transfer** to the other. Classic correlation-based clustering (López de Prado, *Machine Learning for Asset Managers*, ch. 4) cannot see this similarity.

This project builds a dimensionless microstructure signature for each asset, clusters assets on it, and tests whether the clusters predict **strategy transferability**.

## Positioning

| Existing work | How this project differs |
|---|---|
| Correlation clustering (*MLAM*, ch. 4) | Clusters on what happens *inside* the order book, not on price co-movements |
| Commonality in liquidity (Velu et al., §3.4) | Commonality asks whether liquidity *moves together over time*; this project asks whether the *mechanics are the same*, even without co-movement |
| Universal stylized facts (Bouchaud et al.) | Bouchaud emphasizes universality; this project looks for **systematic deviations** from it and tests whether they are exploitable |
| ClusterLOB (arXiv 2504.20349) | Clusters *orders* within one book, not *assets* |

To my knowledge, no published work combines the three steps: (1) a dimensionless signature per asset, (2) ONC clustering on it, and (3) **validation of the clusters by strategy transferability**. Step (3) is what makes the clusters useful rather than merely descriptive.

## Hypotheses (fixed before looking at real data)

- **H1 (structure):** ONC on the signatures finds ≥ 2 clusters with higher quality (t-stat of silhouettes) than on permuted signatures.
- **H2 (difference):** the microstructure partition differs from the returns partition (ARI < 0.3).
- **H3 (usefulness):** a strategy's transfer loss correlates with microstructure distance (Mantel, p < 0.05) **more strongly** than with correlation distance.
- **H4 (stability):** the geometry is stable across periods (correlation of distance matrices > 0.7). Partition ARI is reported too, but it is too noise-sensitive to serve as the criterion.

If H3 fails on real data, the idea is refuted. That is a valid outcome.

## The signature (all quantities are dimensionless)

| Feature | Definition | Reference | What it captures |
|---|---|---|---|
| `gamma` | exponent of C(ℓ) ~ ℓ^-γ for order signs (aggregated variance) | Bouchaud, ch. 10 | metaorder splitting, long memory |
| `c1` | lag-1 autocorrelation of order signs | ch. 10 | short-term herding |
| `log_spread_ticks` | log(effective spread / tick) | §4.8 | large-tick vs small-tick |
| `p_move` | fraction of trades that move the mid | §4.8, Cartea §3.4 | price-grid constraint |
| `impact1` | R(1) / spread, with R(ℓ) = E[ε_t (m_{t+ℓ} − m_t)] | ch. 11 | instantaneous impact |
| `impact_persist` | R(100)/R(1), squashed into (−1, 1) | ch. 13 (propagator) | book resilience |
| `sigma_spread` | per-trade volatility / spread | ch. 16 | volatility/spread balance |
| `vr50` | variance ratio over 50 trades | ch. 2 | intraday mean reversion vs trend |
| `hawkes_n` | branching ratio of an exponential Hawkes process (MLE) | ch. 9 | endogeneity of activity |
| `cancel_to_trade`* | cancelled volume / traded volume | Cartea §4.4 | HFT activity, flickering liquidity |

\* Requires L2 book data, collected with `scripts/collect_binance_depth.py`. Not yet part of the clustering.

All other features are computed from **signed trades only** (price, quantity, aggressor side), which are free and historical on Binance. The mid is approximated as `price − sign × spread/2`.

## Results on synthetic data

`python scripts/run_demo.py` generates 32 simulated assets in 4 known families, using Lillo–Mike–Farmer metaorders, a propagator, a tick grid and Hawkes arrivals.

| Test | Result |
|---|---|
| ONC recovers the families | ARI = 0.84 (5 clusters found for 4 true ones: family D splits on `impact_persist`) |
| Returns partition ≠ micro partition | ARI = 0.04 |
| Mantel: transfer loss ~ micro distance (out-of-sample) | ρ = 0.71, p = 0.0005 |
| Mantel: transfer loss ~ returns distance | ρ = −0.02, p = 0.62 |
| Mean transfer loss within / across families | 0.032 / 0.141 |
| Stability, 1st vs 2nd half | cluster ARI = 0.49, distance correlation = 0.95 |

**These results say nothing about real markets.** In the simulation, microstructure and returns are independent by construction, and microstructure drives the strategy. The demo shows that **the measurement pipeline works**: the estimators recover the parameters, ONC recovers the blocks, and the Mantel test detects the relationship when it exists. This follows the logic of *MLAM* §4.5, which validates ONC on injected blocks.

### Design choices learned along the way

- **Out-of-sample transfer test.** Signatures and calibration use the first half of the data (by time); the loss is measured on the second half. An earlier version used the same data for both, which was circular, since order-sign memory is both a feature and the engine of the test strategy.
- **Two stability measures.** Partition ARI is fragile: a single cluster merge makes it collapse. The correlation between the two periods' distance matrices is far more robust. In the demo, families are stable by construction, yet ARI drops to 0.49 while distances stay correlated at 0.95.
- **Estimating γ.** Fitting the autocorrelation directly is too noisy on 30,000 trades (std ≈ 0.25). The aggregated-variance method is about twice as precise. On real data, the target is **≥ 200,000 trades per asset**.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_demo.py                                  # synthetic demo, ~1 min, no network
python scripts/run_binance.py --start 2026-09-01 --days 1   # quick test on real data
python scripts/run_binance.py --start 2026-09-01 --days 7   # first real study
```

`run_binance.py` downloads aggTrades for 30 pairs from data.binance.vision and caches them in `./data`. It then computes the signatures, runs ONC on the signatures and on hourly returns, the out-of-sample transfer test, a sample-size control and the stability analysis. A first run on 30 spot pairs over one day completed end to end.

## Status and next steps

| Step | Goal | Status |
|---|---|---|
| Synthetic validation | Check that estimators, ONC and the Mantel test behave as expected | ✅ Done |
| Pipeline on real data | Download, signatures, clustering, transfer test on Binance spot | ✅ Runs end to end (1 day) |
| Reality check | 7-day study: do the clusters make economic sense (stablecoins, large-tick pairs, memecoins, majors)? | ⏳ Next |
| Robustness | 4 disjoint periods to test H4; drop-one-feature analysis (MDA/MDI from *MLAM* ch. 6, applied to clustering) | Planned |
| Realistic strategy | Replace the sign-EWMA IC with a cost-paying strategy (Avellaneda–Stoikov market making or order-book imbalance) and measure transfer as net P&L after fees: the decisive test of H3 | Planned |
| L2 data | Collect order-book data and add `cancel_to_trade` and best-level depth relative to median trade size | Planned |

## Possible extensions

- **Tick-size changes as natural experiments.** When an exchange changes a pair's tick, the asset should *migrate* between clusters: a causal validation, not just a correlation (Bouchaud §4.8).
- **Same asset, several venues.** BTC spot, BTC perpetual and BTC on another exchange: if their signatures diverge, the relevant unit is asset × venue, not the asset.
- **Cluster trajectories.** An asset changing cluster from one week to the next reveals a regime change (new market makers, a listing, hype), which may be a signal in itself.
- **Directional transfer.** The loss matrix L(A→B) is not symmetric. "Universal donor" assets, whose calibration works everywhere, are worth identifying.

## Research protocol

Following López de Prado (*MLAM* ch. 8, *Advances in Financial ML* ch. 11–14):

- **Every** configuration tried (features, period, universe) is logged, so that results can be deflated by the number of trials (deflated Sharpe ratio).
- Hypotheses and thresholds are fixed **before** running on real data, as stated above.
- Signatures vary with time of day (intraday seasonality, Cartea ch. 4), so assets are compared over the same hours.
- In crypto, fees and maker rebates depend on account tier: a transfer that is profitable gross may not be net of fees.

## Known limitations

- The mid is approximated from trades only. For very large-tick pairs, `impact1` can turn slightly negative, a bias of the proxy; with L2 data, the true mid should be used.
- The Hawkes model is univariate with an exponential kernel. Real order flow has a power-law kernel (Bouchaud ch. 9), so `hawkes_n` is a comparative measure, not a true value.
- The current test strategy (sign-EWMA IC) ignores costs.
- The L2 collector is written but not yet tested against the live feed.
- Code comments and console output are currently in French.

## Structure

```
microclust/
  simulate.py       synthetic markets with controlled microstructure
  features.py       dimensionless signature (trades only)
  book_features.py  cancellation rate from L2 data (+ self-test)
  onc.py            ONC from MLAM ch. 4, ported to Python 3 and generalized
  transfer.py       transfer loss, Mantel test, stability
  binance.py        download and parse Binance public archives
scripts/
  run_demo.py               synthetic demo + figure
  run_binance.py            real-data study
  collect_binance_depth.py  live L2 collector
results/                    outputs (CSV + figure)
```

## References

- J.-P. Bouchaud, J. Bonart, J. Donier, M. Gould, *Trades, Quotes and Prices*, Cambridge University Press, 2018.
- Á. Cartea, S. Jaimungal, J. Penalva, *Algorithmic and High-Frequency Trading*, Cambridge University Press, 2015.
- M. López de Prado, *Machine Learning for Asset Managers*, Cambridge University Press, 2020.
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
- R. Velu, M. Hardy, D. Nehren, *Algorithmic Trading and Quantitative Strategies*, CRC Press, 2020.

---

<p align="center">
  <a href="https://github.com/matthieu-briche">
    <img src="assets/logo.png" alt="Matthieu Briche" width="37">
  </a>
  <br>
  <sub>Matthieu Briche · <a href="https://github.com/matthieu-briche">github.com/matthieu-briche</a></sub>
</p>

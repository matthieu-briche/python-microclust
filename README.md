microclust: clustering assets by market mechanics, not by returns
Version française
The idea in one sentence
Two assets can be uncorrelated in returns yet work the same way at the microstructure level: same order-sign memory, same price impact, same tick regime, same endogeneity. If so, an intraday strategy calibrated on one should transfer to the other. Classic correlation-based clustering (López de Prado, Machine Learning for Asset Managers, ch. 4) cannot see this similarity.
What is (probably) new
Existing work	How this differs
Correlation clustering (MLAM, ch. 4)	We cluster on what happens inside the order book, not on price co-movements
Commonality in liquidity (Velu et al., §3.4)	Measures whether liquidity moves together over time. Here we measure whether the mechanics are the same, even without co-movement
Universal stylized facts (Bouchaud et al.)	Bouchaud emphasizes universality. This project looks for systematic deviations from it and tests whether they are exploitable
ClusterLOB (arXiv 2504.20349)	Clusters orders within one book, not assets
I have not found published work that (1) builds a dimensionless signature per asset, (2) applies ONC clustering to it, and (3) validates the clusters by strategy transferability. Step (3) is what makes the project useful rather than merely descriptive. Worth re-checking SSRN and arXiv q-fin before investing months.
Testable hypotheses (written BEFORE looking at real data)
H1 (structure): ONC on the signatures finds ≥ 2 clusters with higher quality (t-stat of silhouettes) than on permuted signatures.
H2 (difference): the microstructure partition differs from the returns partition (ARI < 0.3).
H3 (usefulness): a strategy's transfer loss correlates with microstructure distance (Mantel, p < 0.05) more strongly than with correlation distance.
H4 (stability): the geometry is stable across periods (correlation of distance matrices > 0.7). Partition ARI is reported too, but it is too noise-sensitive to serve as the criterion.
If H3 fails on real data, the idea is refuted. That is a valid result, and it only costs a few days.
The signature (all quantities are dimensionless)
Feature	Definition	Reference	What it captures
`gamma`	exponent of C(ℓ) ~ ℓ^-γ for order signs (aggregated variance)	Bouchaud, ch. 10	metaorder splitting, long memory
`c1`	lag-1 autocorrelation of order signs	ch. 10	short-term herding
`log_spread_ticks`	log(effective spread / tick)	§4.8	large-tick vs small-tick
`p_move`	fraction of trades that move the mid	§4.8, Cartea §3.4	price-grid constraint
`impact1`	R(1) / spread, with R(ℓ) = E[ε_t (m_{t+ℓ} − m_t)]	ch. 11	instantaneous impact
`impact_persist`	R(100)/R(1), squashed into (−1, 1)	ch. 13 (propagator)	book resilience
`sigma_spread`	per-trade volatility / spread	ch. 16	volatility/spread balance
`vr50`	variance ratio over 50 trades	ch. 2	intraday mean reversion vs trend
`hawkes_n`	branching ratio of an exponential Hawkes process (MLE)	ch. 9	endogeneity of activity
`cancel_to_trade`*	cancelled volume / traded volume	Cartea §4.4	HFT activity, flickering liquidity
* The last feature needs L2 book data, which you must collect yourself (`scripts/collect_binance_depth.py`). It is not yet part of the clustering.
Everything is computed from signed trades only (price, quantity, aggressor side). This data is free and historical on Binance. The mid is approximated as `price − sign × spread/2`.
Synthetic demo results
`python scripts/run_demo.py` generates 32 simulated assets in 4 known families, using Lillo-Mike-Farmer metaorders, a propagator, a tick grid and Hawkes arrivals.
Test	Result
ONC recovers the families	ARI = 0.84 (5 clusters found for 4 true ones: family D splits on `impact_persist`)
Returns partition ≠ micro partition	ARI = 0.04
Mantel transfer loss ~ micro distance (out-of-sample)	ρ = 0.71, p = 0.0005
Mantel transfer loss ~ returns distance	ρ = −0.02, p = 0.62
Mean transfer loss within / across families	0.032 / 0.141
Stability, 1st vs 2nd half	cluster ARI = 0.49, distance correlation = 0.95
Caution: this proves nothing about real markets. In the simulation, microstructure and returns are independent by construction, and microstructure drives the strategy. The demo only shows that the measurement pipeline works: estimators recover the parameters, ONC recovers the blocks, and the Mantel test detects the relationship when it exists. Same logic as MLAM §4.5, which validates ONC on injected blocks.
The transfer test is out-of-sample: signatures and calibration on the first half (by time), loss measured on the second. An earlier version used the same data for both, which was circular, since order-sign memory is both a feature and the engine of the test strategy.
Stability is measured two ways. Partition ARI is fragile: one cluster merging with another makes it collapse. The correlation between the two periods' distance matrices is much more robust. In the demo, families are stable by construction, yet ARI drops to 0.49 while distances stay correlated at 0.95.
Another lesson learned: estimating γ by directly fitting the autocorrelation is too noisy on 30,000 trades (std ≈ 0.25). It was replaced by the aggregated-variance method, about twice as precise. On real data, aim for ≥ 200,000 trades per asset.
Running on real data
```bash
pip install -r requirements.txt
python scripts/run_demo.py                                  # ~1 min, no network
python scripts/run_binance.py --start 2026-09-01 --days 1   # quick test
python scripts/run_binance.py --start 2026-09-01 --days 7   # first real study
```
The script downloads aggTrades for 30 pairs from data.binance.vision and caches them in `./data`. It then computes the signatures, micro ONC, ONC on hourly returns, the out-of-sample transfer test, a sample-size control and stability. A first real run (30 spot pairs, 1 day) worked end to end.
Suggested roadmap
Week 1: reality check. Run `run_binance.py` over 7 days. Look at `clusters.csv`. Do the clusters make sense? For example: stablecoins and large-tick pairs together, memecoins together, majors together.
Week 2: robustness. Repeat over 4 disjoint periods to test H4. Drop one feature at a time to see which ones drive the clusters, in the spirit of MDA/MDI from MLAM ch. 6 applied to clustering.
Week 3: a real strategy to transfer. Replace the sign-EWMA IC with a strategy that pays costs, e.g. Avellaneda-Stoikov market making (Cartea, ch. 10) or an order-book-imbalance strategy. Measure transfer loss as net P&L after fees. This is where H3 is really decided.
Week 4: collect L2. Run the collector for a few days on a VPS, then add `cancel_to_trade` and best-level depth relative to median trade size.
Most original extensions
Tick-size changes as natural experiments. When an exchange changes a pair's tick, the asset should migrate between clusters. That is causal validation, not just correlation (Bouchaud §4.8 discusses tick changes).
Same asset, several markets. BTC spot, BTC perpetual and BTC on another exchange: if their signatures diverge, the relevant unit is asset × venue, not the asset.
Cluster trajectories. An asset that changes cluster from one week to the next reveals a regime change (new market makers, a listing, hype), which may be a signal in itself.
Transfer is directional. The loss matrix L(A→B) is not symmetric. "Universal donor" assets, whose calibration works everywhere, are worth identifying.
Safeguards (López de Prado, MLAM ch. 8 and Advances in Financial ML ch. 11–14)
Log every configuration tried (features, period, universe). The number of trials is needed to deflate results (deflated Sharpe ratio).
Fix hypotheses and thresholds before running on real data. This README already does.
Signatures vary with time of day (intraday seasonality, Cartea ch. 4). Compare assets over the same hours.
In crypto, fees and maker rebates depend on account tier: a transfer that is profitable gross may not be net of fees.
Known limitations of the prototype
The mid is approximated from trades only. For very large-tick pairs, `impact1` can turn slightly negative, which is a bias of the proxy. With L2 data, the true mid should be used.
The Hawkes model is univariate with an exponential kernel. Real order flow has a power-law kernel (Bouchaud ch. 9), so `hawkes_n` is a comparative measure, not a true value.
The test strategy (sign-EWMA IC) ignores costs.
The L2 collector is written but untested against the live feed.
Structure
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
Code comments and console output are currently in French.

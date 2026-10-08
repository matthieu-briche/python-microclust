"""
Démo de validation sur données SYNTHÉTIQUES (aucune donnée réelle nécessaire).

    python scripts/run_demo.py

Étapes :
  1. univers de 32 actifs répartis en 4 familles de microstructure connues ;
  2. signature de microstructure de chaque actif ;
  3. ONC sur les signatures  -> retrouve-t-on les familles ? (ARI)
  4. ONC sur les corrélations de rendements -> partition différente ?
  5. test de transfert de stratégie (Mantel) ;
  6. stabilité des clusters entre 1re et 2e moitié des données.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from microclust.simulate import make_universe, make_return_panel, FAMILIES
from microclust.features import signature_table, FEATURES
from microclust.onc import onc, standardize, corr_to_obs, labels_from
from microclust.transfer import transfer_loss, euclid, mantel, split_stability

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("1) simulation de l'univers...")
universe, truth = make_universe(per_family=8, n_trades=30_000, seed=42)
ticks = {}  # on laisse le code inférer le tick, comme sur données réelles

print("2) signatures de microstructure...")
S = signature_table(universe, ticks)
S.to_csv(OUT / "signatures.csv")
print(S.groupby(truth).mean().round(3))

print("\n3) ONC sur les signatures")
Z = standardize(S)
cl_micro, silh = onc(Z, seed=0)
lab_micro = labels_from(cl_micro, Z.index)
print(f"   clusters trouvés : {len(cl_micro)}  |  ARI vs vraies familles = {ari(truth, lab_micro):.3f}")

print("\n4) ONC sur les corrélations de rendements journaliers")
R, sector = make_return_panel(list(universe))
cl_ret, _ = onc(corr_to_obs(R.corr()), seed=0)
lab_ret = labels_from(cl_ret, R.columns)
print(f"   clusters trouvés : {len(cl_ret)}  |  ARI vs secteurs = {ari(sector, lab_ret):.3f}"
      f"  |  ARI vs familles microstructure = {ari(truth, lab_ret):.3f}")

print("\n5) test de transfert (stratégie EWMA des signes)")
L = transfer_loss(universe)
Lsym = (L + L.T) / 2
D_micro = euclid(Z)
D_ret = corr_to_obs(R.corr())
r_m, p_m = mantel(Lsym, D_micro)
r_r, p_r = mantel(Lsym, D_ret)
tv = np.asarray(truth.values, dtype=object)
same = tv[:, None] == tv[None, :]
off = ~np.eye(len(truth), dtype=bool)
print(f"   Mantel(perte, distance microstructure) : rho = {r_m:.3f}  p = {p_m:.4f}")
print(f"   Mantel(perte, distance rendements)     : rho = {r_r:.3f}  p = {p_r:.4f}")
print(f"   perte moyenne intra-cluster micro = {Lsym.values[same & off].mean():.4f}"
      f"  | inter = {Lsym.values[~same].mean():.4f}")

print("\n6) stabilité temporelle (ARI 1re moitié vs 2e moitié)")
stab = split_stability(universe, ticks)
print(f"   ARI = {stab:.3f}")

summary = pd.Series({
    "ari_micro_vs_truth": ari(truth, lab_micro), "n_clusters_micro": len(cl_micro),
    "ari_returns_vs_truth": ari(truth, lab_ret), "mantel_rho_micro": r_m, "mantel_p_micro": p_m,
    "mantel_rho_returns": r_r, "mantel_p_returns": p_r, "split_stability_ari": stab})
summary.to_csv(OUT / "summary.csv")

# ------------------------------------------------------------------ figure
BLUE, ORANGE, MID = "#2a78d6", "#eb6834", "#f0efec"
INK, INK2 = "#0b0b0b", "#52514e"
cmap = LinearSegmentedColormap.from_list("div", [BLUE, MID, ORANGE])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})

fig = plt.figure(figsize=(13, 5.2), facecolor="#fcfcfb")
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.55)

ax = fig.add_subplot(gs[0])
order = lab_micro.sort_values(kind="stable").index
im = ax.imshow(Z.loc[order].values, aspect="auto", cmap=cmap, vmin=-3, vmax=3)
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(order)), [f"{n} ({truth[n][0]})" for n in order], fontsize=6)
bounds = np.where(np.diff(lab_micro[order].values) != 0)[0]
for b in bounds:
    ax.axhline(b + 0.5, color=INK, lw=1)
ax.set_title(f"Signatures standardisées, triées par cluster ONC\n(ARI vs vraies familles = {ari(truth, lab_micro):.2f})",
             fontsize=10, loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("z-score robuste")

iu = np.triu_indices(len(truth), 1)
same_u = same[iu]
for k, (D, title, r, p) in enumerate([(D_micro, "distance de microstructure", r_m, p_m),
                                      (D_ret, "distance de corrélation de rendements", r_r, p_r)]):
    ax = fig.add_subplot(gs[k + 1])
    x, y = D.values[iu], Lsym.values[iu]
    ax.scatter(x[~same_u], y[~same_u], s=10, color=ORANGE, alpha=0.55, lw=0, label="familles différentes")
    ax.scatter(x[same_u], y[same_u], s=12, color=BLUE, alpha=0.85, lw=0, marker="D", label="même famille")
    ax.set_xlabel(title)
    ax.set_ylabel("perte de transfert (IC perdu)")
    ax.set_title(f"Perte de transfert vs {title.split(' de ', 1)[1]}\nMantel ρ = {r:.2f}, p = {p:.3f}",
                 fontsize=10, loc="left")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    if k == 0:
        ax.legend(frameon=False, fontsize=8)

fig.suptitle("Démo synthétique — validation de la chaîne, pas un résultat empirique", y=1.04,
             x=0.01, ha="left", fontsize=11, color=INK2)
fig.savefig(OUT / "demo_synthetique.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nfichiers écrits dans {OUT}")

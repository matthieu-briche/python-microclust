# microclust : regrouper les actifs par mécanique de marché, pas par rendements

## L'idée en une phrase

Deux actifs peuvent être **décorrélés en rendements** mais **fonctionner de la même façon** au niveau microstructure : même mémoire des ordres, même impact, même régime de tick, même endogénéité. Si c'est le cas, une stratégie intraday calibrée sur l'un devrait se **transférer** à l'autre. Le clustering classique sur les corrélations (López de Prado, *ML for Asset Managers*, ch. 4) ne peut pas voir cette similarité.

## Ce qui est (probablement) neuf

| Existant | En quoi c'est différent |
|---|---|
| Clustering de corrélations (*MLAM*, ch. 4) | On regroupe sur ce qui se passe *dans* le carnet, pas sur les co-mouvements de prix |
| Commonality in liquidity (Velu et al., §3.4) | Mesure si la liquidité *bouge ensemble dans le temps*. Ici on mesure si la *mécanique est la même*, même sans co-mouvement |
| Faits stylisés universels (Bouchaud et al.) | Bouchaud insiste sur l'universalité. Ce projet cherche les **écarts systématiques** à cette universalité et teste s'ils sont exploitables |
| ClusterLOB (arXiv 2504.20349) | Regroupe des *ordres* dans un carnet, pas des *actifs* |

Je n'ai pas trouvé de travail publié qui (1) construise une signature sans dimension par actif, (2) applique l'ONC dessus et (3) **valide les clusters par la transférabilité d'une stratégie**. C'est l'étape (3) qui rend le projet utile, et pas seulement descriptif. À revérifier sur SSRN et arXiv q-fin avant d'investir des mois.

## Hypothèses testables (à écrire AVANT de regarder les données réelles)

- **H1 (structure)** : l'ONC sur les signatures trouve ≥ 2 clusters avec une qualité (t-stat des silhouettes) supérieure à celle obtenue sur des signatures permutées.
- **H2 (différence)** : la partition microstructure diffère de la partition par rendements, avec un ARI < 0,3.
- **H3 (utilité)** : la perte de transfert d'une stratégie est corrélée à la distance de microstructure (Mantel, p < 0,05) **plus fortement** qu'à la distance de corrélation.
- **H4 (stabilité)** : les clusters sont stables d'une période à l'autre (ARI entre périodes > 0,5).

Si H3 échoue sur données réelles, l'idée est réfutée. C'est un résultat valable, et ça ne coûte que quelques jours.

## La signature (toutes les grandeurs sont sans dimension)

| Feature | Définition | Référence | Ce qu'elle capte |
|---|---|---|---|
| `gamma` | exposant de C(ℓ) ~ ℓ^-γ des signes d'ordres (variance agrégée) | Bouchaud, ch. 10 | découpage des métaordres, mémoire longue |
| `c1` | autocorrélation des signes au retard 1 | ch. 10 | « herding » court terme |
| `log_spread_ticks` | log(spread effectif / tick) | §4.8 | grand tick ↔ petit tick |
| `p_move` | fraction des transactions qui déplacent le mid | §4.8, Cartea §3.4 | contrainte de la grille de prix |
| `impact1` | R(1) / spread, avec R(ℓ) = E[ε_t (m_{t+ℓ} − m_t)] | ch. 11 | impact instantané |
| `impact_persist` | R(100)/R(1), compressé dans (−1, 1) | ch. 13 (propagateur) | résilience du carnet |
| `sigma_spread` | volatilité par transaction / spread | ch. 16 | équilibre volatilité/spread |
| `vr50` | ratio de variance sur 50 transactions | ch. 2 | retour à la moyenne vs tendance intraday |
| `hawkes_n` | ratio de branchement d'un Hawkes exponentiel (MLE) | ch. 9 | endogénéité de l'activité |
| `cancel_to_trade`* | volume annulé / volume exécuté | Cartea §4.4 | activité HFT, « flickering liquidity » |

\* La dernière feature demande le carnet L2, qu'il faut collecter soi-même (`scripts/collect_binance_depth.py`). Elle n'entre pas encore dans le clustering.

Tout est calculé à partir des seuls **trades signés** (prix, quantité, côté de l'agresseur). Ces données sont gratuites et historiques sur Binance. Le mid est approché par `prix − signe × spread/2`.

## Résultats de la démo synthétique

`python scripts/run_demo.py` produit 32 actifs simulés en 4 familles connues, via des métaordres de Lillo-Mike-Farmer, un propagateur, une grille de tick et des arrivées Hawkes.

| Test | Résultat |
|---|---|
| ONC retrouve les familles | ARI = 0,84 (5 clusters trouvés pour 4 vrais : la famille D est coupée en deux sur `impact_persist`) |
| Partition rendements ≠ partition micro | ARI = 0,04 |
| Mantel perte de transfert ~ distance micro | ρ = 0,64, p = 0,0005 |
| Mantel perte de transfert ~ distance rendements | ρ = −0,03, p = 0,76 |
| Perte de transfert moyenne intra- / inter-famille | 0,030 / 0,144 |
| Stabilité 1re / 2e moitié | ARI = 1,00 |

**Attention, ceci ne prouve rien sur le vrai marché.** Dans la simulation, la microstructure et les rendements sont indépendants par construction, et c'est la microstructure qui détermine la stratégie. La démo prouve seulement que **la chaîne de mesure fonctionne** : les estimateurs retrouvent les paramètres, l'ONC retrouve les blocs et le test de Mantel détecte la relation quand elle existe. C'est la même logique que le §4.5 de *MLAM*, où l'on valide l'ONC sur des blocs injectés.

Leçon apprise en route : l'estimation directe de γ par ajustement de l'autocorrélation est trop bruitée sur 30 000 transactions (écart-type ≈ 0,25). Elle a été remplacée par la méthode de la variance agrégée, environ deux fois plus précise. Sur données réelles, vise **≥ 200 000 transactions par actif**.

## Lancer sur données réelles

```bash
pip install -r requirements.txt
python scripts/run_demo.py                                      # 1 min, sans réseau
python scripts/run_binance.py --start 2026-09-01 --days 1 --max-trades 300000   # test rapide
python scripts/run_binance.py --start 2026-09-01 --days 5       # vraie première étude
```

Le script télécharge les aggTrades de 30 paires depuis data.binance.vision et les met en cache dans `./data`. Il calcule ensuite les signatures, l'ONC micro, l'ONC sur les rendements horaires, le test de transfert et la stabilité. Le parseur a été testé hors ligne sur les deux formats d'archives : spot en microsecondes sans en-tête, futures en millisecondes avec en-tête. **Il n'a pas été testé contre le vrai site**, faute d'accès réseau là où le code a été écrit.

## Feuille de route suggérée

1. **Semaine 1 : réalité.** Lancer `run_binance.py` sur 5 jours. Regarder `clusters.csv`. Les clusters ont-ils un sens ? Par exemple : stablecoins et paires à grand tick ensemble, memecoins ensemble, majors ensemble.
2. **Semaine 2 : robustesse.** Refaire l'analyse sur 4 périodes disjointes pour tester H4. Enlever une feature à la fois pour voir lesquelles structurent les clusters. C'est l'esprit du MDA/MDI de *MLAM* ch. 6, appliqué au clustering.
3. **Semaine 3 : une vraie stratégie à transférer.** Remplacer l'IC de l'EWMA des signes par une stratégie avec coûts, par exemple le market making d'Avellaneda-Stoikov (Cartea, ch. 10) ou une stratégie sur le déséquilibre du carnet. Mesurer la perte de transfert en P&L net de frais. C'est là que H3 se joue vraiment.
4. **Semaine 4 : collecter le L2.** Faire tourner le collecteur quelques jours sur un VPS, puis ajouter `cancel_to_trade` et la profondeur au meilleur prix rapportée à la taille médiane des trades.

## Extensions les plus originales

- **Expériences naturelles de changement de tick.** Quand une plateforme change le tick d'une paire, l'actif devrait *migrer* de cluster. C'est une validation causale, pas seulement corrélationnelle (Bouchaud §4.8 discute des changements de tick).
- **Même actif, plusieurs marchés.** BTC spot, BTC perpétuel et BTC sur une autre plateforme : si leurs signatures divergent, l'unité pertinente est le couple actif × plateforme, pas l'actif.
- **Trajectoires de cluster.** Un actif qui change de cluster d'une semaine à l'autre révèle un changement de régime (nouveaux market makers, listing, hype), et c'est potentiellement un signal en soi.
- **Transfert dans les deux sens.** La matrice de perte L(A→B) n'est pas symétrique. Les actifs « donneurs universels », dont la calibration marche partout, sont intéressants à identifier.

## Garde-fous (López de Prado, *MLAM* ch. 8 et *Advances* ch. 11–14)

- Consigner **chaque** configuration essayée (features, période, univers) dans un journal. Le nombre d'essais sert à dégonfler les résultats (Sharpe dégonflé).
- Fixer les hypothèses et les seuils **avant** de lancer sur données réelles. Ce README le fait déjà.
- Les signatures varient selon l'heure (saisonnalité intraday, Cartea ch. 4). Comparer les actifs sur les mêmes fenêtres horaires.
- Sur la crypto, les frais et les rebates maker dépendent du niveau du compte : un transfert « rentable » brut peut ne pas l'être net de frais.

## Limites connues du prototype

- Le mid est approché à partir des trades seulement. Pour les paires à très grand tick, `impact1` peut devenir légèrement négatif, ce qui est un biais du proxy. Avec le carnet L2, il faudra utiliser le vrai mid.
- Le Hawkes est univarié, à noyau exponentiel. Le vrai flux a un noyau en loi de puissance (Bouchaud ch. 9), donc `hawkes_n` sert de mesure comparative, pas de vraie valeur.
- La stratégie témoin (IC de l'EWMA des signes) ne tient pas compte des coûts.
- Le collecteur L2 est écrit mais n'a pas été testé contre le flux réel.

## Structure

```
microclust/
  simulate.py       marchés synthétiques à microstructure contrôlée
  features.py       signature sans dimension (trades seuls)
  book_features.py  taux d'annulation depuis le L2 (+ auto-test)
  onc.py            ONC de MLAM ch. 4, porté en Python 3 et généralisé
  transfer.py       perte de transfert, test de Mantel, stabilité
  binance.py        téléchargement et parsing des archives publiques Binance
scripts/
  run_demo.py               démo synthétique + figure
  run_binance.py            étude sur données réelles
  collect_binance_depth.py  collecteur L2 temps réel
results/                    sorties (CSV + figure)
```

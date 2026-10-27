# Supply Chain Resilience

**Machine Learning-Based Delivery Delay Prediction for Supply Chain Resilience**

Mupasa Kalunga Jean-Luc (UDBL, Lubumbashi) — Prof. Mpyana Mwamba Merlec (Korea University)

## Objet

L'article mesure l'effet du protocole d'évaluation (partition aléatoire contre partition chronologique) sur la performance estimée des modèles de prédiction des retards de livraison, sur deux jeux de données publics :

- **DataCo Smart Supply Chain** — jeu principal ([Mendeley Data](https://doi.org/10.17632/8gx2fvg2k6.5)) ;
- **Olist Brazilian E-Commerce** — validation externe ([Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)).

## Structure

```
data/raw/                  données brutes (non versionnées) : dataco/, olist/
data/processed/            données préparées (non versionnées)
data/predictions/          prédictions des modèles (non versionnées, environ 1,6 Go)
models/                    une étude de réglage par fichier JSON : hyperparamètres, essais, durées
reports/results/           résumés et évaluations (JSON) : source de tous les chiffres de l'article
reports/figures/           figures générées
reports/draft/             article LaTeX ; tables/ contient les tableaux générés
supply_chain_resilience/   code source
tests/                     tests automatiques
```

## Installation

Python 3.11 ; versions figées dans `requirements.txt`.

```bash
conda create -n supply_chain_resilience python=3.11 -y
conda activate supply_chain_resilience
pip install -r requirements.txt
```

Données brutes à placer ainsi :

```
data/raw/dataco/DataCoSupplyChainDataset.csv
data/raw/olist/olist_customers_dataset.csv
data/raw/olist/olist_geolocation_dataset.csv
data/raw/olist/olist_order_items_dataset.csv
data/raw/olist/olist_order_payments_dataset.csv
data/raw/olist/olist_orders_dataset.csv
data/raw/olist/olist_products_dataset.csv
data/raw/olist/olist_sellers_dataset.csv
```

## Reproduire les résultats

Dans l'ordre :

| Commande | Rôle | Durée indicative |
|---|---|---|
| `make data` | cibles, filtres, censure d'Olist, variables d'historique, diagnostics | quelques minutes |
| `make train` | réglage et entraînement : 180 études par jeu (2 protocoles × 2 tâches × 3 modèles × 3 stratégies × 5 graines) | 55 h (Olist) et 87 h (DataCo) de calcul sur un processeur à 4 cœurs |
| `make evaluate` | métriques, écart d'optimisme, profil de dérive, seuil et coût | quelques minutes |
| `make figures` | figures et tableaux de la section 6, en français et en anglais | moins d'une minute |
| `make test` | tests automatiques (fuite, précédence, groupes, budgets de réglage) | environ 7 minutes |

`make train` peut être interrompu et relancé : une étude terminée est sautée, et une étude refaite donne le même résultat (graines fixées). Pour un seul jeu : `python -m supply_chain_resilience.train --dataset olist`.

Sans relancer l'entraînement, `make figures` régénère les tableaux et les figures à partir des fichiers versionnés de `reports/results/`.

L'article se compile dans `reports/draft/` : `latexmk -pdf supply_chain_resilience.tex` pour l'article (anglais), `latexmk -pdf supply_chain_resilience_fr.tex` pour la version de travail en français.

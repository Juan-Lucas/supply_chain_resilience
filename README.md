# Supply Chain Resilience

**Machine Learning-Based Delivery Delay Prediction for Supply Chain Resilience**

Mupasa Kalunga Jean-Luc (UDBL, Lubumbashi) — Prof. Mpyana Mwamba Merlec (Korea University)

## Objet

L'article mesure l'effet du protocole d'évaluation (partition aléatoire contre partition chronologique) sur la performance estimée des modèles de prédiction des retards de livraison, sur deux jeux de données publics :

- **DataCo Smart Supply Chain** — jeu principal ([Mendeley Data](https://doi.org/10.17632/8gx2fvg2k6.5)) ;
- **Olist Brazilian E-Commerce** — validation externe ([Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)).

## Structure

```
data/raw/          données brutes (non versionnées) : dataco/, olist/
reports/draft/     article LaTeX (version de travail : supply_chain_resilience_fr.tex)
reports/figures/   figures générées par le code
supply_chain_resilience/   code source
```

## Démarrage

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Placer les fichiers CSV des deux jeux dans `data/raw/dataco/` et `data/raw/olist/`.

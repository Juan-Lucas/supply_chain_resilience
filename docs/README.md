# Documentation du projet

**Article :** *Machine Learning-Based Delivery Delay Prediction for Supply Chain Resilience*
(version de travail française : *Prédiction des retards de livraison par apprentissage automatique pour la résilience des chaînes d'approvisionnement*)

**Auteurs :** Mupasa Kalunga Jean-Luc (UDBL, Lubumbashi) — Prof. Mpyana Mwamba Merlec (Korea University)
**Mise à jour :** 14/10/2026

---

## 1. En une phrase

L'article ne propose pas un nouvel algorithme : il mesure **combien le protocole d'évaluation (partition aléatoire contre partition chronologique) change la performance estimée** des modèles de prédiction des retards de livraison, sur deux jeux de données publics et indépendants (DataCo, Olist), et ce que cet écart coûte en décisions opérationnelles.

| | |
|---|---|
| **Question de recherche (RQ)** | Dans quelle mesure le protocole d'évaluation influence-t-il la fiabilité estimée des modèles de ML pour la prédiction des retards de livraison, sous évolution temporelle de la distribution des données ? |
| **RQ1** | La dégradation sous évaluation temporelle se retrouve-t-elle sur des jeux réels et indépendants ? |
| **RQ2** | Quelles conséquences pour les décisions opérationnelles de réponse aux retards ? |
| **C1** | Comparaison rigoureuse sans fuite : référence linéaire régularisée, forêt aléatoire, XGBoost (+ Optuna), classification **et** régression |
| **C2** | Quantification de l'écart d'optimisme Δ (aléatoire − chronologique), profil de dérive par trimestre, analyse coût–bénéfice |
| **C3** | Réplication inter-jeux : DataCo (principal) puis Olist (validation externe) |

## 2. Index des documents

| Document | Contenu |
|---|---|
| [`ORGANISATION.md`](ORGANISATION.md) | Analyse de l'état du projet, méthode de travail, découpage FR → EN, questions pour le Prof. Merlec |
| [`PLANNING.md`](PLANNING.md) | Rétroplanning jour par jour jusqu'à l'envoi au Prof. Merlec (dimanche 18/10/2026) |
| [`DICTIONNAIRE_DONNEES.md`](DICTIONNAIRE_DONNEES.md) | Colonnes, cibles, fuites et pièges de DataCo et d'Olist |
| [`../references/LISTE_ARTICLES.md`](../references/LISTE_ARTICLES.md) | Liste de tous les travaux à retrouver (Google Scholar, Crossref…), avec priorité et ce qu'il faut en extraire |
| [`docs/`](docs/) | Site MkDocs (`mkdocs serve`) : accueil et guide d'installation |

## 3. Organisation du dépôt

```
supply_chain_resilience/
├── data/
│   ├── raw/                 <- données brutes, jamais modifiées (non versionnées)
│   │   ├── dataco/          <- DataCoSupplyChainDataset.csv (+ Description…csv)   [À TÉLÉCHARGER]
│   │   └── olist/           <- les 9 CSV Olist                                       [À TÉLÉCHARGER]
│   ├── interim/             <- données nettoyées par jeu
│   └── processed/           <- schéma commun : X, y_clf, y_reg, timestamp, group_id
├── docs/                    <- cette documentation
├── models/                  <- modèles entraînés, études Optuna
├── notebooks/               <- EDA (convention : 1.0-jlm-eda-dataco.ipynb)
├── references/              <- PDF des articles + LISTE_ARTICLES.md
├── reports/
│   ├── draft/               <- article LaTeX
│   │   ├── supply_chain_resilience_fr.tex   <- VERSION DE TRAVAIL (FR)
│   │   ├── related_work_fr.bib              <- bibliographie FR (statuts [OK] / [A VERIFIER])
│   │   ├── supply_chain_resilience.tex      <- squelette EN (traduction ultérieure)
│   │   └── related_work.bib                 <- ancienne bib EN (contient des entrées absentes de la FR)
│   └── figures/             <- figures générées par le code (jamais retouchées à la main)
├── supply_chain_resilience/ <- code source (package Python)
│   ├── config.py            <- chemins
│   ├── dataset.py           <- chargement : load_dataco(), load_olist()           [À ÉCRIRE]
│   ├── features.py          <- caractéristiques admissibles, historique sans fuite  [À ÉCRIRE]
│   ├── modeling/train.py    <- pipelines, partitions, Optuna                        [À ÉCRIRE]
│   ├── modeling/predict.py  <- évaluation, Δ, profil de dérive, coût                [À ÉCRIRE]
│   └── plots.py             <- figures de l'article                                 [À ÉCRIRE]
└── tests/                   <- tests (actuellement un test qui échoue volontairement)
```

## 4. Données

| Jeu | Rôle | Source | Licence |
|---|---|---|---|
| DataCo Smart Supply Chain | Jeu principal | Mendeley Data, DOI 10.17632/8gx2fvg2k6.5 | CC BY 4.0 |
| Olist Brazilian E-Commerce | Validation externe | Kaggle `olistbr/brazilian-ecommerce` | CC BY-NC-SA 4.0 |

Détails : [`DICTIONNAIRE_DONNEES.md`](DICTIONNAIRE_DONNEES.md).

## 5. Pipeline expérimental visé

1. **Chargement** par jeu → **schéma commun** (`X`, `y_clf`, `y_reg`, `timestamp`, `group_id`).
2. **Filtre des fuites** (contrainte de disponibilité, §3.2 de l'article) : instant de prédiction = date de commande.
3. **Deux partitions** sur le même jeu :
   - aléatoire groupée par commande (`GroupShuffleSplit`) ;
   - chronologique avec date de coupure τ* (entraînement = livrées avant τ*, test = commandées après τ*).
4. **Modèles** dans un `Pipeline` scikit-learn ajusté sur l'entraînement seul : régression logistique / linéaire régularisée, RF, XGBoost ; Optuna (TPE) avec plis chronologiques à fenêtre croissante.
5. **Mesures** : PR-AUC, F1, ROC-AUC (classification) ; MAE, RMSE, R² (régression) ; moyenne ± écart-type sur plusieurs graines.
6. **Analyses** : écart Δ par métrique, profil de dérive trimestriel (avec prévalence par trimestre), coût C(θ) sous hypothèses illustratives.
7. Même code, mêmes réglages sur DataCo puis Olist.

## 6. Règles non négociables

1. Même préparation, même protocole, mêmes métriques sur les deux jeux réels.
2. Toute colonne postérieure à la date de commande est une fuite.
3. Le seuil de décision et les hyperparamètres se règlent sur la validation, **jamais** sur le test.
4. Aucune référence citée sans avoir été lue au moins (résumé + méthode + conclusion) et sans métadonnées vérifiées sur Crossref.
5. Toute figure et tout tableau de l'article sont produits par un script du dépôt (reproductibilité).

## 7. Commandes utiles

```bash
# Environnement
python -m venv .venv && .venv/Scripts/activate      # Windows : .venv\Scripts\activate
make requirements                                    # ou : pip install -r requirements.txt

# Compiler l'article FR (depuis reports/draft/)
pdflatex supply_chain_resilience_fr && bibtex supply_chain_resilience_fr \
  && pdflatex supply_chain_resilience_fr && pdflatex supply_chain_resilience_fr

# Documentation
mkdocs serve -f docs/mkdocs.yml
```

Dépendances à ajouter à `requirements.txt` quand le code sera écrit : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `matplotlib`, `scipy`, `shap` (optionnel), `imbalanced-learn` (optionnel, Olist).

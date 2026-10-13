# Mise en route

## 1. Environnement

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Git Bash : source .venv/Scripts/activate)
pip install -r requirements.txt
pip install pandas numpy scikit-learn xgboost optuna matplotlib scipy
```

## 2. Données (non versionnées)

| Jeu | Où le télécharger | Dossier |
|---|---|---|
| DataCo Smart Supply Chain | Mendeley Data, DOI 10.17632/8gx2fvg2k6.5 | `data/raw/dataco/` |
| Olist Brazilian E-Commerce | Kaggle, `olistbr/brazilian-ecommerce` | `data/raw/olist/` |

DataCo se lit avec `encoding="latin-1"`. Noter la date de téléchargement et la version : elles sont citées dans l'article.

## 3. Article

```bash
cd reports/draft
pdflatex supply_chain_resilience_fr
bibtex supply_chain_resilience_fr
pdflatex supply_chain_resilience_fr
pdflatex supply_chain_resilience_fr
```

## 4. Documentation

```bash
mkdocs serve -f docs/mkdocs.yml
```

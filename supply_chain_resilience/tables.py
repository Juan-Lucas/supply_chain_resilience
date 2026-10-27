"""Tableaux de la section 6 : reports/results/<jeu>_evaluation.json -> reports/draft/tables/*.tex.

Aucun chiffre des résultats n'est recopié à la main : l'article inclut ces fragments
avec \\input. Stratégie de réglage TPE (Optuna) sauf pour le tableau des stratégies ;
moyenne sur les graines, avec l'écart-type ou l'intervalle à 95 % de l'éq. ic-delta.
"""

import json

from loguru import logger
import pandas as pd
import typer

from supply_chain_resilience.config import REPORTS_DIR
from supply_chain_resilience.preprocessing import RESULTS_DIR, Dataset

app = typer.Typer()

TABLES_DIR = REPORTS_DIR / "draft" / "tables"
STRATEGY = "tpe"
MAIN_COST_RATIO = 5
DATASET_LABEL = {Dataset.dataco: "DataCo", Dataset.olist: "Olist"}
MODEL_LABEL = {"linear": "Linéaire", "rf": "Forêt aléatoire", "xgb": "XGBoost"}
STRATEGY_LABEL = {"grid": "Grille", "random": "Aléatoire", "tpe": "TPE"}


def num(x: float, digits: int = 3) -> str:
    """Nombre au format français, signe moins typographique."""
    s = f"{abs(x):.{digits}f}".replace(".", "{,}")
    return f"$-${s}" if x < 0 and float(f"{abs(x):.{digits}f}") != 0 else s


def mean_sd(values: pd.Series, digits: int = 3) -> str:
    return f"{num(values.mean(), digits)} $\\pm$ {num(values.std(ddof=1), digits)}"


def gap_cell(row: pd.Series, digits: int = 3) -> str:
    star = "" if row["excludes_zero"] else "\\textsuperscript{ns}"
    return (
        f"{num(row['delta_mean'], digits)} [{num(row['ci_low'], digits)} ; "
        f"{num(row['ci_high'], digits)}]{star}"
    )


def load(dataset: Dataset) -> dict:
    return json.loads(
        (RESULTS_DIR / f"{dataset.value}_evaluation.json").read_text(encoding="utf-8")
    )


def frame(dataset: Dataset, key: str) -> pd.DataFrame:
    return pd.DataFrame(load(dataset)[key]).assign(dataset=dataset)


# ---------------------------------------------------------------------------


def models_table() -> str:
    """Test chronologique (conditions de déploiement) et entraînement, par modèle."""
    lines = []
    for dataset in Dataset:
        m = frame(dataset, "metrics")
        m = m[(m["strategy"] == STRATEGY) & (m["protocol"] == "chrono")]
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{DATASET_LABEL[dataset]}}}}} \\\\")
        for model, label in MODEL_LABEL.items():
            c = m[(m["task"] == "clf") & (m["model"] == model)]
            r = m[(m["task"] == "reg") & (m["model"] == model)]
            ct, cr = c[c["set"] == "test"], c[c["set"] == "train"]
            rt, rr = r[r["set"] == "test"], r[r["set"] == "train"]
            lines.append(
                f"{label} & {mean_sd(ct['pr_auc'])} & {mean_sd(ct['roc_auc'])} & {num(cr['pr_auc'].mean())}"
                f" & {mean_sd(rt['mae'], 2)} & {mean_sd(rt['r2'])} & {num(rr['mae'].mean(), 2)} \\\\"
            )
        lines.append("\\midrule")
    lines.pop()
    return "\n".join(lines) + "\n"


def gaps_table() -> str:
    """Delta [IC 95 %] sur le test, avec la prévalence et le retard moyen de chaque test."""
    lines = []
    for dataset in Dataset:
        g = frame(dataset, "optimism_gaps")
        g = g[g["strategy"] == STRATEGY].set_index(["model", "metric"])
        m = frame(dataset, "metrics")
        m = m[(m["strategy"] == STRATEGY) & (m["set"] == "test")]
        prev = m[m["task"] == "clf"].groupby("protocol")["prevalence"].mean()
        delay = m[m["task"] == "reg"].groupby("protocol")["delay_mean"].mean()
        context = (
            f"prévalence du test {num(100 * prev['chrono'], 1)}\\,\\% (chrono) et "
            f"{num(100 * prev['random'], 1)}\\,\\% (aléatoire) ; retard moyen des commandes en retard "
            f"{num(delay['chrono'], 2)} et {num(delay['random'], 2)} jours"
        )
        lines.append(
            f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{DATASET_LABEL[dataset]}}} -- {context}}} \\\\"
        )
        for model, label in MODEL_LABEL.items():
            cells = [
                gap_cell(g.loc[(model, k)], d)
                for k, d in (("pr_auc", 3), ("roc_auc", 3), ("mae", 3))
            ]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    return "\n".join(lines) + "\n"


def decision_table() -> str:
    """Scénario principal r = 5 : F1 et économie sous chaque protocole, Delta_S [IC 95 %]."""
    lines = []
    for dataset in Dataset:
        d = frame(dataset, "decisions")
        d = d[(d["strategy"] == STRATEGY) & (d["cost_ratio"] == MAIN_COST_RATIO)]
        g = frame(dataset, "decision_gaps")
        g = g[(g["strategy"] == STRATEGY) & (g["cost_ratio"] == MAIN_COST_RATIO)]
        g = g.set_index(["model", "metric"])
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{DATASET_LABEL[dataset]}}}}} \\\\")
        for model, label in MODEL_LABEL.items():
            s = d[d["model"] == model].groupby("protocol")[["f1", "saving"]].mean()
            lines.append(
                f"{label} & {num(s.loc['chrono', 'f1'])} & {num(s.loc['random', 'f1'])}"
                f" & {num(s.loc['chrono', 'saving'])} & {num(s.loc['random', 'saving'])}"
                f" & {gap_cell(g.loc[(model, 'saving')])} \\\\"
            )
        lines.append("\\midrule")
    lines.pop()
    return "\n".join(lines) + "\n"


def strategies_table() -> str:
    """Test chronologique selon la stratégie de réglage, à budget égal."""
    lines = []
    for dataset in Dataset:
        m = frame(dataset, "metrics")
        m = m[(m["protocol"] == "chrono") & (m["set"] == "test")]
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{DATASET_LABEL[dataset]}}}}} \\\\")
        for model, label in MODEL_LABEL.items():
            clf = m[(m["task"] == "clf") & (m["model"] == model)].groupby("strategy")["pr_auc"]
            reg = m[(m["task"] == "reg") & (m["model"] == model)].groupby("strategy")["mae"]
            cells = [num(clf.mean()[s]) for s in STRATEGY_LABEL] + [
                num(reg.mean()[s], 2) for s in STRATEGY_LABEL
            ]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.pop()
    return "\n".join(lines) + "\n"


@app.command()
def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in [
        ("modeles", models_table),
        ("ecart", gaps_table),
        ("decision", decision_table),
        ("strategies", strategies_table),
    ]:
        path = TABLES_DIR / f"{name}.tex"
        header = "% Généré par supply_chain_resilience/tables.py : ne pas modifier à la main.\n"
        path.write_text(header + build(), encoding="utf-8")
        logger.info(path)


if __name__ == "__main__":
    app()

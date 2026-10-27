"""Phase 5, évaluation : métriques, écart d'optimisme et profil de dérive.

Lit les prédictions de train.py (data/predictions/<jeu>/<étude>.parquet) et écrit
reports/results/<jeu>_evaluation.json, source des chiffres des résultats de l'article :
- métriques de chaque étude sur l'entraînement et sur le test (section 3.5) ;
- écart d'optimisme Delta par graine, sa moyenne et son intervalle à 95 % (éq. ic-delta) ;
- profil de dérive mensuel sur le test chronologique (section 3.3) ;
- seuil d'alerte choisi sur les plis de validation regroupés, puis F1 et coût sur le
  test pour chaque rapport de coûts r (sections 3.6 et 4.6).
"""

import itertools
import json

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
import typer

from supply_chain_resilience.config import MODELS_DIR
from supply_chain_resilience.models import Model, Task
from supply_chain_resilience.preprocessing import RESULTS_DIR, Dataset
from supply_chain_resilience.splits import Protocol
from supply_chain_resilience.train import PREDICTIONS_DIR, SEEDS, Strategy, study_name

app = typer.Typer()

# Métrique -> True si une valeur plus grande est meilleure
CLF_METRICS = {"pr_auc": True, "roc_auc": True}
REG_METRICS = {"mae": False, "rmse": False, "r2": True, "smape": False}
HIGHER_IS_BETTER = {**CLF_METRICS, **REG_METRICS}
CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------


def smape(y_true, y_pred) -> float:
    """sMAPE en %, sur des retards strictement positifs (section 3.5)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)))


def classification_metrics(y_true, score) -> dict:
    y_true = np.asarray(y_true, int)
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def regression_metrics(y_true, pred) -> dict:
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
        "r2": float(r2_score(y_true, pred)),
        "smape": smape(y_true, pred),
    }


def set_metrics(rows: pd.DataFrame, task: Task) -> dict:
    """Classification : toutes les lignes. Régression conditionnelle : les retards seuls."""
    if task is Task.clf:
        out = classification_metrics(rows["y_clf"], rows["pred"])
        return {**out, "n": len(rows), "prevalence": float(rows["y_clf"].mean())}
    late = rows[rows["y_clf"] == 1]
    return {
        **regression_metrics(late["y_reg"], late["pred"]),
        "n": len(late),
        # la MAE dépend de la distribution du retard : à rapporter avec elle
        "delay_mean": float(late["y_reg"].mean()),
        "delay_median": float(late["y_reg"].median()),
    }


def load_predictions(dataset: Dataset, name: str) -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_DIR / dataset.value / f"{name}.parquet")


def metrics_table(dataset: Dataset, seeds=SEEDS) -> pd.DataFrame:
    """Une ligne par (étude, ensemble), en colonnes les métriques."""
    records = []
    plan = itertools.product(seeds, Protocol, Task, Model, Strategy)
    for seed, protocol, task, model, strategy in plan:
        pred = load_predictions(dataset, study_name(protocol, seed, task, model, strategy))
        for name in ("train", "test"):
            records.append(
                {
                    "protocol": protocol.value,
                    "seed": seed,
                    "task": task.value,
                    "model": model.value,
                    "strategy": strategy.value,
                    "set": name,
                    **set_metrics(pred[pred["set"] == name], task),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Écart d'optimisme (éq. delta-opt et ic-delta)
# ---------------------------------------------------------------------------


def optimism_gap(chrono: float, random: float, metric: str) -> float:
    """Positif quand le protocole aléatoire paraît meilleur que le chronologique."""
    if HIGHER_IS_BETTER[metric]:
        return random - chrono
    return chrono - random


def confidence_interval(values) -> tuple[float, float, float, float]:
    """Moyenne, écart-type et intervalle de Student sur les graines."""
    values = np.asarray(values, float)
    mean, sd = values.mean(), values.std(ddof=1)
    half = stats.t.ppf(0.5 + CONFIDENCE / 2, len(values) - 1) * sd / np.sqrt(len(values))
    return float(mean), float(sd), float(mean - half), float(mean + half)


def paired_gaps(table: pd.DataFrame, keys: list[str], metrics_of) -> pd.DataFrame:
    """Delta par groupe et par métrique, apparié par graine (même graine, deux protocoles)."""
    rows = []
    for key, group in table.groupby(keys):
        key = dict(zip(keys, key))
        wide = group.pivot(index="seed", columns="protocol")
        for metric in metrics_of(key):
            chrono, random = wide[(metric, "chrono")], wide[(metric, "random")]
            gaps = [optimism_gap(c, r, metric) for c, r in zip(chrono, random)]
            mean, sd, low, high = confidence_interval(gaps)
            rows.append(
                {
                    **key,
                    "metric": metric,
                    "chrono_mean": float(chrono.mean()),
                    "random_mean": float(random.mean()),
                    "delta_by_seed": [float(g) for g in gaps],
                    "delta_mean": mean,
                    "delta_sd": sd,
                    "ci_low": low,
                    "ci_high": high,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(rows)


def optimism_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Delta sur le test, par (tâche, modèle, stratégie, métrique)."""
    test = table[table["set"] == "test"]
    return paired_gaps(
        test,
        ["task", "model", "strategy"],
        lambda key: CLF_METRICS if key["task"] == Task.clf.value else REG_METRICS,
    )


# ---------------------------------------------------------------------------
# Seuil d'alerte et coût (sections 3.6 et 4.6)
# ---------------------------------------------------------------------------


def best_threshold(y_true, score, cost_ratio: float) -> float:
    """Seuil qui minimise C = r FN + FP (c_FP = 1) ; alerte si score >= seuil.

    Les seuils candidats sont les scores observés, plus +inf (aucune alerte).
    """
    y_true, score = np.asarray(y_true, int), np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y_true[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]  # dernier rang de chaque score : ex aequo ensemble
    costs = cost_ratio * (y.sum() - tp[last]) + fp[last]
    candidates = s[last]
    no_alert = cost_ratio * y.sum()
    if no_alert <= costs.min():
        return float("inf")
    return float(candidates[np.argmin(costs)])


def decision_metrics(y_true, score, threshold: float, cost_ratio: float) -> dict:
    """Effet du seuil sur le test : qualité des alertes et coût pour 1 000 expéditions."""
    y_true = np.asarray(y_true, int)
    alert = np.asarray(score, float) >= threshold
    tp = int((alert & (y_true == 1)).sum())
    fp = int((alert & (y_true == 0)).sum())
    fn = int((~alert & (y_true == 1)).sum())
    n, positives = len(y_true), int(y_true.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    cost = cost_ratio * fn + fp
    trivial = min(cost_ratio * positives, n - positives)  # ne jamais alerter, ou toujours
    return {
        "threshold": threshold if np.isfinite(threshold) else None,
        "never_alert": not np.isfinite(threshold),  # la validation dit de ne jamais alerter
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": float(alert.mean()),
        "cost_per_1000": 1000 * cost / n,
        "trivial_cost_per_1000": 1000 * trivial / n,
        "saving": 1 - cost / trivial if trivial else 0.0,
    }


COST_RATIOS = (1, 2, 5, 10)
DECISION_METRICS = {
    "f1": True,
    "precision": True,
    "recall": True,
    "cost_per_1000": False,
    "saving": True,
}
HIGHER_IS_BETTER.update(DECISION_METRICS)


def decision_table(dataset: Dataset, seeds=SEEDS) -> pd.DataFrame:
    """Seuil choisi sur les K plis de validation regroupés, appliqué au test (option B)."""
    records = []
    for seed, protocol, model, strategy in itertools.product(seeds, Protocol, Model, Strategy):
        pred = load_predictions(dataset, study_name(protocol, seed, Task.clf, model, strategy))
        valid, test = pred[pred["set"] == "valid"], pred[pred["set"] == "test"]
        for r in COST_RATIOS:
            threshold = best_threshold(valid["y_clf"], valid["pred"], r)
            records.append(
                {
                    "protocol": protocol.value,
                    "seed": seed,
                    "model": model.value,
                    "strategy": strategy.value,
                    "cost_ratio": r,
                    **decision_metrics(test["y_clf"], test["pred"], threshold, r),
                }
            )
    return pd.DataFrame(records)


def decision_gaps(decisions: pd.DataFrame) -> pd.DataFrame:
    return paired_gaps(
        decisions, ["model", "strategy", "cost_ratio"], lambda key: DECISION_METRICS
    )


# ---------------------------------------------------------------------------
# Profil de dérive (section 3.3)
# ---------------------------------------------------------------------------


def monthly_profile(pred: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Métriques du test chronologique, mois par mois (mois de la commande)."""
    test = pred[pred["set"] == "test"]
    rows = []
    for month, group in test.groupby(test["ordered_at"].dt.to_period("M")):
        if task is Task.clf and group["y_clf"].nunique() < 2:
            continue  # la PR-AUC n'est pas définie sans les deux classes
        if task is Task.reg and (group["y_clf"] == 1).sum() < 2:
            continue
        rows.append({"month": str(month), **set_metrics(group, task)})
    return pd.DataFrame(rows)


def drift_profiles(dataset: Dataset, seeds=SEEDS) -> pd.DataFrame:
    """Profil moyen sur les graines, par (tâche, modèle, stratégie)."""
    frames = []
    plan = itertools.product(seeds, Task, Model, Strategy)
    for seed, task, model, strategy in plan:
        name = study_name(Protocol.chrono, seed, task, model, strategy)
        profile = monthly_profile(load_predictions(dataset, name), task)
        frames.append(
            profile.assign(seed=seed, task=task.value, model=model.value, strategy=strategy.value)
        )
    profiles = pd.concat(frames, ignore_index=True)
    keys = ["task", "model", "strategy", "month"]
    return profiles.drop(columns="seed").groupby(keys, as_index=False).mean()


# ---------------------------------------------------------------------------
# Sortie
# ---------------------------------------------------------------------------


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def compute_summary(dataset: Dataset) -> dict:
    """Durée de calcul cumulée des études (réglage et réajustements), d'après models/<jeu>."""
    studies = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((MODELS_DIR / dataset.value).glob("*.json"))
    ]
    seconds = sum(s["tuning_seconds"] + s.get("refit_seconds", 0) for s in studies)
    return {
        "n_studies": len(studies),
        "n_trials_per_study": sorted({s["n_trials"] for s in studies}),
        "hours": round(seconds / 3600, 1),
    }


@app.command()
def main(dataset: Dataset = typer.Option(..., help="Jeu de données")):
    table = metrics_table(dataset)
    gaps = optimism_gaps(table)
    profiles = drift_profiles(dataset)
    decisions = decision_table(dataset)
    summary = {
        "dataset": dataset.value,
        "seeds": SEEDS,
        "confidence": CONFIDENCE,
        "cost_ratios": list(COST_RATIOS),
        "compute": compute_summary(dataset),
        "metrics": records(table),
        "optimism_gaps": records(gaps),
        "drift_profiles": records(profiles),
        "decisions": records(decisions),
        "decision_gaps": records(decision_gaps(decisions)),
    }
    out = RESULTS_DIR / f"{dataset.value}_evaluation.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"{out} : {len(table)} lignes de métriques, {len(gaps)} écarts")


if __name__ == "__main__":
    app()

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
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
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
# Courbes précision-rappel et ROC sur le test
# ---------------------------------------------------------------------------

CURVE_GRID = np.linspace(0, 1, 101)
CURVE_STRATEGY = Strategy.tpe


def pr_curve(y_true, score, grid=CURVE_GRID) -> np.ndarray:
    """Précision interpolée : au rappel r, la meilleure précision atteinte à un rappel >= r."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, grid, side="left").clip(max=len(recall) - 1)
    return envelope[idx]


def roc_curve_on_grid(y_true, score, grid=CURVE_GRID) -> np.ndarray:
    """Taux de vrais positifs en fonction du taux de faux positifs, sur la grille."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return np.interp(grid, fpr, tpr)


def curve_table(dataset: Dataset, seeds=SEEDS) -> pd.DataFrame:
    """Courbes moyennes sur les graines, par protocole et modèle (classification, TPE)."""
    rows = []
    for protocol, model in itertools.product(Protocol, Model):
        pr, roc, prevalence = [], [], []
        for seed in seeds:
            pred = load_predictions(
                dataset, study_name(protocol, seed, Task.clf, model, CURVE_STRATEGY)
            )
            test = pred[pred["set"] == "test"]
            pr.append(pr_curve(test["y_clf"], test["pred"]))
            roc.append(roc_curve_on_grid(test["y_clf"], test["pred"]))
            prevalence.append(test["y_clf"].mean())
        rows.append(
            {
                "protocol": protocol.value,
                "model": model.value,
                "strategy": CURVE_STRATEGY.value,
                "prevalence": float(np.mean(prevalence)),
                "grid": [float(g) for g in CURVE_GRID],
                "pr_precision": [float(v) for v in np.mean(pr, axis=0)],
                "roc_tpr": [float(v) for v in np.mean(roc, axis=0)],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustesse de l'écart : bootstrap par semaines et prévalence égale
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 200
N_PREVALENCE_DRAWS = 20


def week_blocks(test: pd.DataFrame) -> list[np.ndarray]:
    """Positions des lignes de chaque semaine de commande (blocs du bootstrap)."""
    weeks = test["ordered_at"].dt.to_period("W").to_numpy()
    return [np.flatnonzero(weeks == w) for w in pd.unique(weeks)]


def block_bootstrap(test: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    """PR-AUC et ROC-AUC sur n rééchantillonnages des semaines (colonnes : pr, roc)."""
    blocks = week_blocks(test)
    y, s = test["y_clf"].to_numpy(int), test["pred"].to_numpy(float)
    out = np.empty((n, 2))
    for b in range(n):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        out[b] = average_precision_score(y[idx], s[idx]), roc_auc_score(y[idx], s[idx])
    return out


def matched_prevalence_pr_auc(
    test: pd.DataFrame, prevalence: float, rng: np.random.Generator, n: int
) -> float:
    """PR-AUC moyenne après sous-échantillonnage des retards jusqu'à la prévalence visée."""
    y, s = test["y_clf"].to_numpy(int), test["pred"].to_numpy(float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(prevalence / (1 - prevalence) * len(neg)))
    if n_pos >= len(pos):
        return float(average_precision_score(y, s))
    values = []
    for _ in range(n):
        idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
        values.append(average_precision_score(y[idx], s[idx]))
    return float(np.mean(values))


def robustness_table(dataset: Dataset, seeds=SEEDS) -> pd.DataFrame:
    """Par modèle (classification, TPE) :
    - intervalle à 95 % de l'écart moyen sur les graines, chaque test étant rééchantillonné
      par semaines : il ajoute la variabilité de l'échantillon de test à celle des graines ;
    - PR-AUC du test aléatoire ramené à la prévalence du test chronologique."""
    rows = []
    for model in Model:
        boot = {"pr": [], "roc": []}
        matched, chrono_pr = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tests = {}
            for protocol in Protocol:
                pred = load_predictions(
                    dataset, study_name(protocol, seed, Task.clf, model, CURVE_STRATEGY)
                )
                tests[protocol] = pred[pred["set"] == "test"]
            chrono = block_bootstrap(tests[Protocol.chrono], rng, N_BOOTSTRAP)
            random = block_bootstrap(tests[Protocol.random], rng, N_BOOTSTRAP)
            boot["pr"].append(random[:, 0] - chrono[:, 0])
            boot["roc"].append(random[:, 1] - chrono[:, 1])
            target = tests[Protocol.chrono]["y_clf"].mean()
            matched.append(
                matched_prevalence_pr_auc(tests[Protocol.random], target, rng, N_PREVALENCE_DRAWS)
            )
            chrono_pr.append(
                average_precision_score(
                    tests[Protocol.chrono]["y_clf"], tests[Protocol.chrono]["pred"]
                )
            )
        row = {"model": model.value, "strategy": CURVE_STRATEGY.value, "n_bootstrap": N_BOOTSTRAP}
        for metric, gaps in boot.items():
            mean_over_seeds = np.mean(gaps, axis=0)  # une valeur par réplication
            row[f"delta_{metric}_auc_boot_low"] = float(np.percentile(mean_over_seeds, 2.5))
            row[f"delta_{metric}_auc_boot_high"] = float(np.percentile(mean_over_seeds, 97.5))
        gaps_matched = [m - c for m, c in zip(matched, chrono_pr)]
        mean, sd, low, high = confidence_interval(gaps_matched)
        row.update(
            {
                "random_pr_auc_matched": float(np.mean(matched)),
                "chrono_pr_auc": float(np.mean(chrono_pr)),
                "delta_pr_auc_matched": mean,
                "delta_pr_auc_matched_low": low,
                "delta_pr_auc_matched_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
        "test_curves": records(curve_table(dataset)),
        "robustness": records(robustness_table(dataset)),
        "decisions": records(decisions),
        "decision_gaps": records(decision_gaps(decisions)),
    }
    out = RESULTS_DIR / f"{dataset.value}_evaluation.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"{out} : {len(table)} lignes de métriques, {len(gaps)} écarts")


if __name__ == "__main__":
    app()

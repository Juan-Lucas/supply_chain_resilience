"""Phase 4, réglage et entraînement : une étude par (jeu, protocole, graine, tâche, modèle, stratégie).

Pour chaque étude : réglage sur les K plis de l'entraînement du protocole (section 3.7),
avec B essais quelle que soit la stratégie (grille, recherche aléatoire, TPE) ; puis
ajustement sur tout l'entraînement avec les meilleurs hyperparamètres, et prédictions
sur l'entraînement et sur le test. Les métriques sont calculées en phase 5 (evaluate.py).

Sorties :
- models/<jeu>/<étude>.json : hyperparamètres retenus, essais, durées ;
- data/predictions/<jeu>/<étude>.parquet : prédictions (ignoré par git).
Une étude déjà faite est sautée, ce qui permet de reprendre un calcul interrompu.
"""

from dataclasses import dataclass
from enum import Enum
import itertools
import json
import math
import time

from loguru import logger
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error
import typer

from supply_chain_resilience.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from supply_chain_resilience.models import (
    HUBER_LOSS,
    TARGET,
    Model,
    Task,
    fit_pipeline,
    make_model_pipeline,
    predict,
    select_features,
    task_mask,
)
from supply_chain_resilience.preprocessing import ORDER_DATE, Dataset
from supply_chain_resilience.splits import Protocol, protocol_split, tuning_folds

app = typer.Typer()

PREDICTIONS_DIR = DATA_DIR / "predictions"

N_TRIALS = 30
SEEDS = [0, 1, 2, 3, 4]


class Strategy(str, Enum):
    grid = "grid"
    random = "random"
    tpe = "tpe"


# ---------------------------------------------------------------------------
# Espaces de recherche
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """Hyperparamètre : intervalle pour la recherche aléatoire et TPE, valeurs pour la grille.

    grid=None : absent de la grille, qui le laisse à sa valeur par défaut. Le produit des
    tailles de grille vaut N_TRIALS, pour que les trois stratégies aient le même budget.
    """

    name: str
    low: float
    high: float
    grid: tuple | None
    log: bool = False
    integer: bool = False


RF_SPACE = (
    Param("max_depth", 4, 32, (4, 6, 8, 12, 20, 32), log=True, integer=True),
    Param("min_samples_leaf", 1, 100, (1, 3, 10, 30, 100), log=True, integer=True),
)

XGB_SPACE = (
    Param("learning_rate", 0.01, 0.3, (0.05, 0.2), log=True),
    Param("max_depth", 3, 10, (3, 6, 9), integer=True),
    Param("n_estimators", 100, 1000, (100, 200, 400, 700, 1000), log=True, integer=True),
    Param("subsample", 0.5, 1.0, None),
    Param("colsample_bytree", 0.5, 1.0, None),
    Param("min_child_weight", 1, 50, None, log=True),
    Param("reg_lambda", 1e-3, 10, None, log=True),
)


def log_grid(low: float, high: float, n: int) -> tuple:
    return tuple(float(v) for v in np.logspace(math.log10(low), math.log10(high), n))


def search_space(dataset: Dataset, model: Model, task: Task) -> tuple[Param, ...]:
    if model is Model.rf:
        return RF_SPACE
    if model is Model.xgb:
        return XGB_SPACE
    if task is Task.clf:
        return (Param("C", 1e-4, 1e2, log_grid(1e-4, 1e2, N_TRIALS), log=True),)
    if HUBER_LOSS[dataset]:
        return (
            Param("alpha", 1e-6, 1.0, log_grid(1e-6, 1.0, 6), log=True),
            Param("epsilon", 1.1, 3.0, (1.1, 1.35, 1.75, 2.25, 3.0)),
        )
    return (Param("alpha", 1e-3, 1e3, log_grid(1e-3, 1e3, N_TRIALS), log=True),)


def grid_size(space: tuple[Param, ...]) -> int:
    return math.prod(len(p.grid) for p in space if p.grid is not None)


def suggest(trial: optuna.Trial, space: tuple[Param, ...], strategy: Strategy) -> dict:
    params = {}
    for p in space:
        if strategy is Strategy.grid and p.grid is None:
            continue
        if p.integer:
            params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
        else:
            params[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log)
    return params


def make_sampler(strategy: Strategy, space: tuple[Param, ...], seed: int):
    if strategy is Strategy.grid:
        grid = {p.name: list(p.grid) for p in space if p.grid is not None}
        return optuna.samplers.GridSampler(grid, seed=seed)
    if strategy is Strategy.random:
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)  # 10 essais de démarrage aléatoires


# ---------------------------------------------------------------------------
# Une étude
# ---------------------------------------------------------------------------


def score(y_true, pred, task: Task) -> float:
    """Critère de réglage : PR-AUC (à maximiser) ou MAE (à minimiser)."""
    if task is Task.clf:
        return float(average_precision_score(y_true, pred))
    return float(mean_absolute_error(y_true, pred))


def study_name(protocol: Protocol, seed: int, task: Task, model: Model, strategy: Strategy) -> str:
    return f"{protocol.value}_s{seed}_{task.value}_{model.value}_{strategy.value}"


def run_study(
    df: pd.DataFrame,
    dataset: Dataset,
    protocol: Protocol,
    seed: int,
    task: Task,
    model: Model,
    strategy: Strategy,
    n_trials: int = N_TRIALS,
) -> tuple[dict, pd.DataFrame]:
    """Réglage sur les plis de l'entraînement, puis ajustement final et prédictions."""
    split = protocol_split(df, protocol, seed)
    train = df[split == "train"].reset_index()  # « index » garde la ligne d'origine
    test = df[split == "test"].reset_index()
    folds = tuning_folds(train, dataset, protocol, seed)
    in_task = task_mask(train, task).to_numpy()
    X_train, y_train = select_features(train, dataset), train[TARGET[task]]

    space = search_space(dataset, model, task)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space, strategy)
        values = []
        for fit_rows, valid_rows in folds:
            fit_rows, valid_rows = fit_rows[in_task[fit_rows]], valid_rows[in_task[valid_rows]]
            pipe = make_model_pipeline(dataset, model, task, params, seed)
            fit_pipeline(pipe, X_train.iloc[fit_rows], y_train.iloc[fit_rows], task)
            pred = predict(pipe, X_train.iloc[valid_rows], task)
            values.append(score(y_train.iloc[valid_rows], pred, task))
        trial.set_user_attr("fold_values", values)
        return float(np.mean(values))

    direction = "maximize" if task is Task.clf else "minimize"
    study = optuna.create_study(direction=direction, sampler=make_sampler(strategy, space, seed))
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    tuning_seconds = time.perf_counter() - start

    # Ajustement final sur tout l'entraînement de la tâche
    start = time.perf_counter()
    pipe = make_model_pipeline(dataset, model, task, study.best_params, seed)
    fit_pipeline(pipe, X_train[in_task], y_train[in_task], task)
    fit_seconds = time.perf_counter() - start

    # Régression : prédite sur tout le test, pour combiner ensuite avec le classifieur
    train_rows = train[in_task]
    parts = [
        ("train", train_rows, predict(pipe, X_train[in_task], task)),
        ("test", test, predict(pipe, select_features(test, dataset), task)),
    ]
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "row": rows["index"].to_numpy(),
                    "group_id": rows["group_id"].to_numpy(),
                    "ordered_at": rows[ORDER_DATE[dataset]].to_numpy(),
                    "set": name,
                    "y_clf": rows["y_clf"].to_numpy(),
                    "y_reg": rows["y_reg"].to_numpy(),
                    "pred": pred,
                }
            )
            for name, rows, pred in parts
        ],
        ignore_index=True,
    )

    record = {
        "dataset": dataset.value,
        "protocol": protocol.value,
        "seed": seed,
        "task": task.value,
        "model": model.value,
        "strategy": strategy.value,
        "criterion": "average_precision" if task is Task.clf else "mae",
        "n_trials": n_trials,
        "n_train_rows": int(in_task.sum()),
        "n_test_rows": len(test),
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_fold_values": study.best_trial.user_attrs["fold_values"],
        "tuning_seconds": round(tuning_seconds, 1),
        "final_fit_seconds": round(fit_seconds, 1),
        "trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "fold_values": t.user_attrs["fold_values"],
                "seconds": round(t.duration.total_seconds(), 2),
            }
            for t in study.trials
        ],
    }
    return record, predictions


# ---------------------------------------------------------------------------
# Plan complet
# ---------------------------------------------------------------------------


@app.command()
def main(
    dataset: Dataset = typer.Option(..., help="Jeu de données"),
    seeds: list[int] = typer.Option(SEEDS, "--seed"),
    protocols: list[Protocol] = typer.Option(list(Protocol), "--protocol"),
    tasks: list[Task] = typer.Option(list(Task), "--task"),
    models: list[Model] = typer.Option(list(Model), "--model"),
    strategies: list[Strategy] = typer.Option(list(Strategy), "--strategy"),
    n_trials: int = typer.Option(N_TRIALS, help="Budget d'essais par étude"),
    max_orders: int = typer.Option(
        0, help="Essai rapide : nombre de commandes tirées (0 = toutes)"
    ),
    force: bool = typer.Option(False, help="Refaire les études déjà faites"),
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"{dataset.value}_features.parquet")
    suffix = ""
    if max_orders:
        groups = df["group_id"].drop_duplicates().sample(max_orders, random_state=0)
        df = df[df["group_id"].isin(groups)].reset_index(drop=True)
        suffix = f"_smoke{max_orders}"
    models_dir = MODELS_DIR / f"{dataset.value}{suffix}"
    predictions_dir = PREDICTIONS_DIR / f"{dataset.value}{suffix}"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    plan = list(itertools.product(seeds, protocols, tasks, models, strategies))
    for i, (seed, protocol, task, model, strategy) in enumerate(plan, start=1):
        name = study_name(protocol, seed, task, model, strategy)
        out = models_dir / f"{name}.json"
        if out.exists() and not force:
            continue
        record, predictions = run_study(
            df, dataset, protocol, seed, task, model, strategy, n_trials
        )
        predictions.to_parquet(predictions_dir / f"{name}.parquet", index=False)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"[{i}/{len(plan)}] {dataset.value} {name} : {record['best_value']:.4f} "
            f"({record['tuning_seconds']:.0f} s)"
        )


if __name__ == "__main__":
    app()

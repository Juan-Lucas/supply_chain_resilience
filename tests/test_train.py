"""Contrôle du réglage : budget égal des trois stratégies, étude complète sur un petit échantillon."""

import itertools

import optuna
import pandas as pd
import pytest

from supply_chain_resilience.config import PROCESSED_DATA_DIR
from supply_chain_resilience.models import Model, Task
from supply_chain_resilience.preprocessing import Dataset
from supply_chain_resilience.splits import Protocol, protocol_split
from supply_chain_resilience.train import (
    N_TRIALS,
    Strategy,
    grid_size,
    run_study,
    search_space,
)

COMBOS = list(itertools.product(Dataset, Model, Task))


@pytest.mark.parametrize("dataset,model,task", COMBOS)
def test_grid_has_exactly_the_common_budget(dataset, model, task):
    assert grid_size(search_space(dataset, model, task)) == N_TRIALS


@pytest.mark.parametrize("dataset,model,task", COMBOS)
def test_grid_values_lie_in_the_search_intervals(dataset, model, task):
    for p in search_space(dataset, model, task):
        for v in p.grid or ():
            assert p.low <= v <= p.high, (p.name, v)


@pytest.fixture(scope="module")
def small_olist():
    path = PROCESSED_DATA_DIR / "olist_features.parquet"
    if not path.exists():
        pytest.skip("lancer make data d'abord")
    df = pd.read_parquet(path)
    groups = df["group_id"].drop_duplicates().sample(6000, random_state=0)
    return df[df["group_id"].isin(groups)].reset_index(drop=True)


@pytest.fixture(autouse=True)
def quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("protocol", list(Protocol))
@pytest.mark.parametrize("task", list(Task))
def test_study_predicts_only_its_own_test_rows(small_olist, protocol, task):
    df = small_olist
    record, pred = run_study(
        df, Dataset.olist, protocol, 0, task, Model.xgb, Strategy.random, n_trials=2
    )
    split = protocol_split(df, protocol, 0)
    test_rows = set(df.index[split == "test"])
    train_rows = set(df.index[split == "train"])
    assert set(pred.loc[pred["set"] == "test", "row"]) == test_rows
    assert set(pred.loc[pred["set"] == "train", "row"]) <= train_rows
    if task is Task.reg:  # entraînement conditionnel aux retards
        assert (pred.loc[pred["set"] == "train", "y_clf"] == 1).all()
    assert record["n_trials"] == len(record["trials"]) == 2
    assert len(record["best_fold_values"]) == 4


@pytest.mark.parametrize("strategy", list(Strategy))
def test_same_seed_gives_same_study(small_olist, strategy):
    args = (small_olist, Dataset.olist, Protocol.random, 1, Task.clf, Model.linear, strategy)
    a, pa = run_study(*args, n_trials=3)
    b, pb = run_study(*args, n_trials=3)
    assert [t["params"] for t in a["trials"]] == [t["params"] for t in b["trials"]]
    pd.testing.assert_series_equal(pa["pred"], pb["pred"])


def test_grid_trials_are_grid_points(small_olist):
    record, _ = run_study(
        small_olist, Dataset.olist, Protocol.chrono, 0, Task.clf, Model.rf, Strategy.grid, 3
    )
    space = {p.name: p.grid for p in search_space(Dataset.olist, Model.rf, Task.clf)}
    for t in record["trials"]:
        for name, value in t["params"].items():
            assert value in space[name]

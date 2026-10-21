"""Contrôle des partitions : précédence des plis chronologiques, groupes, partition aléatoire."""

import numpy as np
import pandas as pd
import pytest

from supply_chain_resilience.config import PROCESSED_DATA_DIR
from supply_chain_resilience.preprocessing import ORDER_DATE, Dataset
from supply_chain_resilience.splits import (
    K_FOLDS,
    Protocol,
    chrono_fold_dates,
    chrono_folds,
    order_table,
    random_folds,
    random_split,
    tuning_folds,
)


def features_path(dataset):
    return PROCESSED_DATA_DIR / f"{dataset.value}_features.parquet"


needs_data = pytest.mark.skipif(
    not all(features_path(d).exists() for d in Dataset),
    reason="lancer make data d'abord",
)


@pytest.fixture(scope="module")
def frames():
    return {d: pd.read_parquet(features_path(d)) for d in Dataset}


def toy_train(n_orders=200, seed=0):
    """Commandes Olist fictives, une ligne par commande, livrées 1 à 30 jours après."""
    rng = np.random.default_rng(seed)
    ordered = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 300, n_orders)), "D"
    )
    return pd.DataFrame(
        {
            "group_id": [f"o{i}" for i in range(n_orders)],
            ORDER_DATE[Dataset.olist]: ordered,
            "delivered_at": ordered + pd.to_timedelta(rng.integers(1, 30, n_orders), "D"),
        }
    )


# ---------------------------------------------------------------------------
# Plis chronologiques
# ---------------------------------------------------------------------------


def check_chrono_folds(train, dataset):
    orders = order_table(train, dataset)
    dates = chrono_fold_dates(train, dataset)
    folds = chrono_folds(train, dataset)
    assert len(folds) == K_FOLDS
    for j, (fit_rows, valid_rows) in enumerate(folds):
        fit = orders.loc[train["group_id"].iloc[fit_rows].unique()]
        valid = orders.loc[train["group_id"].iloc[valid_rows].unique()]
        # précédence : tout ce qui sert à ajuster est livré avant la première commande validée
        assert fit["delivered_at"].max() <= dates[j] < valid["ordered_at"].min()
        if j + 1 < K_FOLDS:
            assert valid["ordered_at"].max() <= dates[j + 1]
        assert not set(fit.index) & set(valid.index)
    # fenêtre croissante
    for (a, _), (b, _) in zip(folds, folds[1:]):
        assert set(a) <= set(b)
    # les validations successives ne se recouvrent pas
    valid_all = np.concatenate([v for _, v in folds])
    assert len(valid_all) == len(set(valid_all))
    return folds


def test_chrono_folds_respect_precedence_on_toy_orders():
    train = toy_train()
    folds = check_chrono_folds(train, Dataset.olist)
    # une commande passée avant s_j mais livrée après n'est ni ajustée ni validée au pli j
    orders = order_table(train, Dataset.olist)
    s1 = chrono_fold_dates(train, Dataset.olist)[0]
    straddling = orders[(orders["ordered_at"] <= s1) & (orders["delivered_at"] > s1)].index
    assert len(straddling) > 0
    used = set(train["group_id"].iloc[np.concatenate(folds[0])])
    assert not used & set(straddling)


def test_fold_dates_are_quantiles_of_one_date_per_order():
    train = toy_train()
    doubled = pd.concat([train, train.iloc[:50]])  # lignes répétées d'une même commande
    pd.testing.assert_series_equal(
        chrono_fold_dates(train, Dataset.olist), chrono_fold_dates(doubled, Dataset.olist)
    )


@needs_data
@pytest.mark.parametrize("dataset", list(Dataset))
def test_chrono_folds_on_real_training_set(frames, dataset):
    df = frames[dataset]
    train = df[df["split_chrono"] == "train"].reset_index(drop=True)
    folds = check_chrono_folds(train, dataset)
    assert all(len(f) > 0 and len(v) > 0 for f, v in folds)


# ---------------------------------------------------------------------------
# Plis aléatoires groupés
# ---------------------------------------------------------------------------


@needs_data
def test_random_folds_keep_order_lines_together(frames):
    df = frames[Dataset.dataco]  # plusieurs lignes par commande
    train = df[df["split_chrono"] == "train"].reset_index(drop=True)
    folds = random_folds(train, seed=0)
    assert len(folds) == K_FOLDS
    valid_all = np.concatenate([v for _, v in folds])
    assert sorted(valid_all) == list(range(len(train)))
    for fit_rows, valid_rows in folds:
        assert not set(train["group_id"].iloc[fit_rows]) & set(train["group_id"].iloc[valid_rows])
    other = random_folds(train, seed=1)
    assert not np.array_equal(folds[0][1], other[0][1])


def test_tuning_folds_follow_the_protocol():
    train = toy_train()
    chrono = tuning_folds(train, Dataset.olist, Protocol.chrono, seed=0)
    for (a, b), (c, d) in zip(chrono, chrono_folds(train, Dataset.olist)):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    rand = tuning_folds(train, Dataset.olist, Protocol.random, seed=0)
    assert sum(len(v) for _, v in rand) == len(train)


# ---------------------------------------------------------------------------
# Partition aléatoire
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("dataset", list(Dataset))
def test_random_split_uses_same_orders_and_test_size(frames, dataset):
    df = frames[dataset]
    split = random_split(df, seed=0)
    per_order = pd.DataFrame({"chrono": df["split_chrono"], "random": split, "g": df["group_id"]})
    per_order = per_order.groupby("g").agg(["first", "nunique"])
    assert (per_order[("random", "nunique")] == 1).all()  # commandes entières
    chrono, rand = per_order[("chrono", "first")], per_order[("random", "first")]
    assert ((chrono == "gap") == (rand == "gap")).all()
    assert (rand == "test").sum() == (chrono == "test").sum()
    assert random_split(df, seed=0).equals(split)
    assert not random_split(df, seed=1).equals(split)

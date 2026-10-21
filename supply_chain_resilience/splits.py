"""Phase 4, partitions : partition aléatoire groupée et plis de réglage des deux protocoles.

La partition chronologique est déjà dans split_chrono (preprocessing.add_splits). La
partition aléatoire porte sur les mêmes commandes (train et test chronologiques), avec un
test de même nombre de commandes ; les commandes « gap » restent écartées (section 5).
Toutes les lignes d'une commande vont dans le même ensemble (contrainte de groupes).
"""

from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from supply_chain_resilience.preprocessing import ORDER_DATE, Dataset


class Protocol(str, Enum):
    chrono = "chrono"
    random = "random"


K_FOLDS = 4

Folds = list[tuple[np.ndarray, np.ndarray]]


def order_table(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Une ligne par commande : date de commande et date de livraison de la dernière ligne."""
    return df.groupby("group_id").agg(
        ordered_at=(ORDER_DATE[dataset], "first"), delivered_at=("delivered_at", "max")
    )


def random_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Tirage uniforme des commandes de test parmi train et test chronologiques."""
    orders = df.groupby("group_id")["split_chrono"].first()
    kept = orders[orders != "gap"]
    n_test = int((kept == "test").sum())
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(kept.index.to_numpy(), size=n_test, replace=False)
    split = pd.Series("train", index=kept.index)
    split[test_ids] = "test"
    split = split.reindex(orders.index, fill_value="gap")
    return df["group_id"].map(split).rename("split_random")


def protocol_split(df: pd.DataFrame, protocol: Protocol, seed: int) -> pd.Series:
    if protocol is Protocol.chrono:
        return df["split_chrono"]
    return random_split(df, seed)


def chrono_fold_dates(train: pd.DataFrame, dataset: Dataset, k: int = K_FOLDS) -> pd.Series:
    """s_1 < ... < s_K : quantiles d'ordre j/(K+1) des dates de commande (une par commande)."""
    ordered_at = order_table(train, dataset)["ordered_at"]
    return ordered_at.quantile([j / (k + 1) for j in range(1, k + 1)]).reset_index(drop=True)


def chrono_folds(train: pd.DataFrame, dataset: Dataset, k: int = K_FOLDS) -> Folds:
    """Plis à fenêtre croissante (éq. plis) : T_j = livrées au plus tard à s_j,
    V_j = commandées dans ]s_j, s_{j+1}], avec s_{K+1} = tau* (toutes les commandes
    de l'entraînement sont passées avant tau*)."""
    orders = order_table(train, dataset)
    dates = list(chrono_fold_dates(train, dataset, k)) + [pd.Timestamp.max]
    row_orders = orders.loc[train["group_id"]]
    ordered_at = row_orders["ordered_at"].to_numpy()
    delivered_at = row_orders["delivered_at"].to_numpy()
    folds = []
    for j in range(k):
        s, s_next = np.datetime64(dates[j]), np.datetime64(dates[j + 1])
        fit_rows = np.flatnonzero(delivered_at <= s)
        valid_rows = np.flatnonzero((ordered_at > s) & (ordered_at <= s_next))
        folds.append((fit_rows, valid_rows))
    return folds


def random_folds(train: pd.DataFrame, seed: int, k: int = K_FOLDS) -> Folds:
    """K plis tirés au hasard, groupés par commande."""
    splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(train, groups=train["group_id"]))


def tuning_folds(train: pd.DataFrame, dataset: Dataset, protocol: Protocol, seed: int) -> Folds:
    """Chaque protocole est validé selon son propre principe (section 5)."""
    if protocol is Protocol.chrono:
        return chrono_folds(train, dataset)
    return random_folds(train, seed)

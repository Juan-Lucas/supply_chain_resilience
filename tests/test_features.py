"""Contrôle anti-fuite des variables d'historique : petit jeu construit à la main, puis données réelles."""

import numpy as np
import pandas as pd
import pytest

from supply_chain_resilience.config import PROCESSED_DATA_DIR
from supply_chain_resilience.features import HISTORY_ENTITIES, add_history_features, history_rate
from supply_chain_resilience.preprocessing import ORDER_DATE, Dataset


def make_orders(rows):
    """rows : (entité, jour de commande, jour de livraison, y_clf)."""
    base = pd.Timestamp("2018-01-01")
    return pd.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "ordered_at": [base + pd.Timedelta(days=r[1]) for r in rows],
            "delivered_at": [base + pd.Timedelta(days=r[2]) for r in rows],
            "y_clf": [r[3] for r in rows],
        }
    )


def test_only_orders_delivered_before_prediction_count():
    orders = make_orders(
        [
            ("A", 0, 5, 1),  # livrée le jour 5
            ("A", 1, 20, 1),  # encore en transit au jour 10 : ne doit pas compter
            ("A", 10, 15, 0),  # commande évaluée au jour 10
        ]
    )
    out = history_rate(orders, ("entity",), window_days=90, min_orders=1)
    assert out.loc[2, "n"] == 1
    assert out.loc[2, "rate"] == 1.0


def test_order_never_counts_in_its_own_history():
    # livrée le jour même de sa commande (DataCo : 0 jour réel)
    orders = make_orders([("A", 3, 3, 1), ("A", 3, 3, 0)])
    out = history_rate(orders, ("entity",), window_days=90, min_orders=1)
    assert out["n"].tolist() == [1, 1]
    assert out["rate"].tolist() == [0.0, 1.0]


def test_window_and_entities_are_respected():
    orders = make_orders(
        [
            ("A", 0, 1, 1),  # hors fenêtre de 30 jours au jour 40
            ("A", 30, 35, 0),
            ("B", 30, 35, 1),  # autre entité
            ("A", 40, 45, 0),
        ]
    )
    out = history_rate(orders, ("entity",), window_days=30, min_orders=1)
    assert out.loc[3, "n"] == 1
    assert out.loc[3, "rate"] == 0.0


def test_rate_missing_below_min_orders():
    orders = make_orders([("A", 0, 1, 1), ("A", 1, 2, 1), ("A", 10, 12, 0)])
    out = history_rate(orders, ("entity",), window_days=90, min_orders=3)
    assert out.loc[2, "n"] == 2
    assert np.isnan(out.loc[2, "rate"])


def test_matches_brute_force_on_random_orders():
    rng = np.random.default_rng(0)
    size = 300
    ordered = rng.integers(0, 200, size)
    rows = [(rng.choice(["A", "B", "C"]), int(o), int(o + rng.integers(0, 15)), int(rng.integers(0, 2))) for o in ordered]
    orders = make_orders(rows)
    out = history_rate(orders, ("entity",), window_days=30, min_orders=5)

    for i, row in orders.iterrows():
        mask = (
            (orders.index != i)
            & (orders["entity"] == row["entity"])
            & (orders["delivered_at"] <= row["ordered_at"])
            & (orders["delivered_at"] >= row["ordered_at"] - pd.Timedelta(days=30))
        )
        assert out.loc[i, "n"] == mask.sum()
        expected = orders.loc[mask, "y_clf"].mean() if mask.sum() >= 5 else np.nan
        assert out.loc[i, "rate"] == pytest.approx(expected, nan_ok=True)


# ---------------------------------------------------------------------------
# Données réelles : data/processed/<jeu>.parquet, produit par make data
# ---------------------------------------------------------------------------

def load_processed(dataset: Dataset) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{dataset.value}.parquet"
    if not path.exists():
        pytest.skip(f"{path} absent : lancer make data")
    return pd.read_parquet(path)


@pytest.fixture(scope="module", params=[Dataset.dataco, Dataset.olist], ids=["dataco", "olist"])
def enriched(request):
    dataset = request.param
    df = add_history_features(load_processed(dataset), dataset)
    return dataset, df


def test_history_columns_are_valid(enriched):
    dataset, df = enriched
    for name in HISTORY_ENTITIES[dataset]:
        assert (df[f"hist_{name}_n"] >= 0).all()
        assert df[f"hist_{name}_rate"].dropna().between(0, 1).all()
        # même valeur sur toutes les lignes d'une commande
        assert (df.groupby("group_id")[f"hist_{name}_rate"].nunique(dropna=False) == 1).all()


def test_history_uses_only_past_deliveries_on_real_orders(enriched):
    """Recalcul direct de la définition (section 3.2) pour 200 commandes tirées au hasard."""
    dataset, df = enriched
    orders = df.drop_duplicates("group_id")
    sample = orders.sample(200, random_state=0)
    window = pd.Timedelta(days=90)
    order_date = ORDER_DATE[dataset]

    for name, cols in HISTORY_ENTITIES[dataset].items():
        for _, row in sample.iterrows():
            same_entity = np.logical_and.reduce([orders[c] == row[c] for c in cols])
            mask = (
                same_entity
                & (orders["group_id"] != row["group_id"])
                & (orders["delivered_at"] <= row[order_date])
                & (orders["delivered_at"] >= row[order_date] - window)
            )
            assert row[f"hist_{name}_n"] == mask.sum()
            expected = orders.loc[mask, "y_clf"].mean() if mask.sum() >= 10 else np.nan
            assert row[f"hist_{name}_rate"] == pytest.approx(expected, nan_ok=True)


def test_olist_history_rates_are_informative(enriched):
    """Phase 3 : la prévalence augmente avec le taux historique de l'État et de la liaison."""
    dataset, df = enriched
    if dataset is not Dataset.olist:
        pytest.skip("contrôle propre à Olist")
    orders = df.drop_duplicates("group_id")
    for name in ("customer_state", "route"):
        quartile = pd.qcut(orders[f"hist_{name}_rate"], 4, labels=False)
        prevalence = orders.groupby(quartile)["y_clf"].mean()
        assert prevalence.iloc[-1] > 2 * prevalence.iloc[0]

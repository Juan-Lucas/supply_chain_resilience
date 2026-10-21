"""Contrôle de preprocessing.py : les chiffres de référence viennent de notebooks/01_eda.ipynb."""

import pandas as pd
import pytest

from supply_chain_resilience.config import RAW_DATA_DIR
from supply_chain_resilience.preprocessing import (
    Dataset,
    add_date_features,
    add_splits,
    aggregate_olist,
    apply_filters,
    build_targets,
    drop_excluded_columns,
    haversine_km,
    load_dataco,
    load_olist,
)

needs_raw_data = pytest.mark.skipif(
    not (RAW_DATA_DIR / "dataco").exists() or not (RAW_DATA_DIR / "olist").exists(),
    reason="données brutes absentes de data/raw",
)


def run_pipeline(dataset: Dataset) -> tuple[pd.DataFrame, dict]:
    """Même enchaînement que main, sans écrire de fichier."""
    summary: dict = {}
    if dataset is Dataset.dataco:
        df = apply_filters(build_targets(load_dataco(), dataset), dataset, summary)
    else:
        tables = load_olist()
        df = apply_filters(build_targets(tables["orders"], dataset), dataset, summary)
        df = aggregate_olist(df, tables)
    df = add_splits(add_date_features(drop_excluded_columns(df, dataset), dataset), dataset, summary)
    return df, summary


@pytest.fixture(scope="module")
def dataco():
    return run_pipeline(Dataset.dataco)


@pytest.fixture(scope="module")
def olist():
    return run_pipeline(Dataset.olist)


# ---------------------------------------------------------------------------
# Fonctions élémentaires
# ---------------------------------------------------------------------------


def test_haversine_known_distance():
    # São Paulo - Rio de Janeiro : environ 360 km à vol d'oiseau
    assert haversine_km(-23.55, -46.63, -22.91, -43.17) == pytest.approx(361, abs=5)
    assert haversine_km(-23.55, -46.63, -23.55, -46.63) == 0


def test_olist_targets_use_calendar_days():
    orders = pd.DataFrame(
        {
            "order_delivered_customer_date": pd.to_datetime(["2018-01-10 23:59", "2018-01-10 00:01", None]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-10", "2018-01-09", "2018-01-09"]),
        }
    )
    out = build_targets(orders, Dataset.olist)
    assert out["delta"].tolist()[:2] == [0, 1]
    assert out["y_clf"].tolist()[:2] == [0, 1]
    assert out["delta"].isna().iloc[2]


# ---------------------------------------------------------------------------
# DataCo
# ---------------------------------------------------------------------------


@needs_raw_data
def test_dataco_filters(dataco):
    df, summary = dataco
    assert summary["n_raw"] == 180519
    assert summary["n_canceled"] == 7754
    assert len(df) == 172765
    assert df["group_id"].nunique() == 62897


@needs_raw_data
def test_dataco_target_and_prevalence(dataco):
    df, _ = dataco
    per_order = df.drop_duplicates("group_id")
    assert per_order["y_clf"].mean() == pytest.approx(0.5731, abs=1e-4)
    assert df["delta"].between(-2, 4).all()


@needs_raw_data
def test_dataco_no_leakage_columns(dataco):
    df, _ = dataco
    for col in ["Days for shipping (real)", "shipping date (DateOrders)", "Delivery Status", "Late_delivery_risk", "Order Status"]:
        assert col not in df.columns


@needs_raw_data
def test_dataco_chrono_split(dataco):
    df, summary = dataco
    assert summary["chrono_cut"] == "2017-05-12 10:42:00"
    assert (summary["n_orders_train"], summary["n_orders_test"]) == (47172, 15725)
    assert (df.groupby("group_id")["split_chrono"].nunique() == 1).all()


# ---------------------------------------------------------------------------
# Olist
# ---------------------------------------------------------------------------


@needs_raw_data
def test_olist_censoring(olist):
    _, summary = olist
    assert summary["end_of_observation"] == "2018-09-03 17:40:06"
    assert summary["censoring_horizon_days"] == 29
    assert summary["censoring_cutoff"] == "2018-08-05"
    assert summary["n_before_first_year"] == 267
    assert summary["n_censored"] == 5209


@needs_raw_data
def test_olist_one_row_per_order(olist):
    df, _ = olist
    assert len(df) == 90994
    assert df["order_id"].is_unique
    assert df["y_clf"].mean() == pytest.approx(0.0687, abs=1e-4)


@needs_raw_data
def test_olist_aggregation(olist):
    df, _ = olist
    assert (df["n_sellers"] > 1).sum() == 1180
    assert df["distance_km"].isna().sum() == 453
    assert df["distance_km"].median() == pytest.approx(440.2, abs=0.1)
    assert not (df["total_weight_g"] == 0).any()
    assert not (df["payment_installments"] == 0).any()


@needs_raw_data
def test_olist_no_post_purchase_columns(olist):
    df, _ = olist
    for col in ["order_status", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "shipping_limit_date"]:
        assert col not in df.columns


@needs_raw_data
def test_olist_chrono_split(olist):
    _, summary = olist
    assert summary["chrono_cut"].startswith("2018-04-19 12:45")
    assert (summary["n_orders_train"], summary["n_orders_test"]) == (68245, 22749)
    assert summary["prevalence_train"] == pytest.approx(0.0781, abs=1e-4)
    assert summary["prevalence_test"] == pytest.approx(0.0404, abs=1e-4)

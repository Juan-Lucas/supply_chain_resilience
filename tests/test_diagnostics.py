"""Contrôle de diagnostics.py : valeurs de référence de notebooks/01_eda.ipynb."""

import pytest

from supply_chain_resilience.config import RAW_DATA_DIR
from supply_chain_resilience.diagnostics import (
    dataco_cancellations,
    dataco_rupture,
    olist_censoring,
    olist_shipping_limit,
)
from supply_chain_resilience.preprocessing import load_dataco, load_olist

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_DIR / "dataco").exists() or not (RAW_DATA_DIR / "olist").exists(),
    reason="données brutes absentes de data/raw",
)


@pytest.fixture(scope="module")
def dataco():
    return load_dataco()


@pytest.fixture(scope="module")
def olist():
    return load_olist()


def test_dataco_cancellations(dataco):
    out = dataco_cancellations(dataco)
    assert out["n_canceled_lines"] == 7754
    assert out["late_delivery_risk_values"] == [0]
    assert out["n_indicator_disagreements"] == 4423


def test_dataco_rupture(dataco):
    out = dataco_rupture(dataco)
    assert out["products_per_month_before"] == [42, 87]
    assert out["products_per_month_after"] == [8, 20]
    assert out["lines_per_order_after"][1] < 1.1 < out["lines_per_order_before"][0]


def test_olist_shipping_limit(olist):
    out = olist_shipping_limit(olist)
    assert out["share_same_time_as_approval"] == pytest.approx(0.783, abs=1e-3)
    assert out["share_same_time_as_purchase"] == pytest.approx(0.035, abs=1e-3)
    assert out["most_frequent_days_after_approval"] == 6.0


def test_olist_censoring(olist):
    out = olist_censoring(olist)
    last = list(out["last_weeks"].values())
    assert last[-2] == pytest.approx(5.04, abs=0.01)
    assert last[-1] == pytest.approx(1.79, abs=0.01)

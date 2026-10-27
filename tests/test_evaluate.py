"""Contrôle de l'évaluation : métriques, signe de l'écart d'optimisme, intervalle, profil mensuel."""

import numpy as np
import pandas as pd
import pytest

from supply_chain_resilience.evaluate import (
    confidence_interval,
    monthly_profile,
    optimism_gap,
    optimism_gaps,
    set_metrics,
    smape,
)
from supply_chain_resilience.models import Task


def test_perfect_classifier_scores_one():
    rows = pd.DataFrame({"y_clf": [0, 0, 1, 1], "y_reg": 0, "pred": [0.1, 0.2, 0.8, 0.9]})
    out = set_metrics(rows, Task.clf)
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["prevalence"] == 0.5 and out["n"] == 4


def test_regression_metrics_use_late_rows_only():
    rows = pd.DataFrame(
        {"y_clf": [0, 1, 1], "y_reg": [0, 2, 4], "pred": [100.0, 3.0, 3.0]}
    )  # la ligne à l'heure, très mal prédite, ne doit pas compter
    out = set_metrics(rows, Task.reg)
    assert out["n"] == 2
    assert out["mae"] == pytest.approx(1.0)
    assert out["rmse"] == pytest.approx(1.0)


def test_smape_formula():
    assert smape([2, 4], [2, 4]) == 0
    assert smape([1], [3]) == pytest.approx(100 * 2 / 2)


def test_gap_is_positive_when_random_protocol_looks_better():
    assert optimism_gap(chrono=0.10, random=0.26, metric="pr_auc") == pytest.approx(0.16)
    assert optimism_gap(chrono=7.0, random=6.0, metric="mae") == pytest.approx(1.0)
    assert optimism_gap(chrono=0.5, random=0.4, metric="r2") == pytest.approx(-0.1)


def test_confidence_interval_matches_student_with_five_seeds():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, sd, low, high = confidence_interval(values)
    assert mean == 3.0
    assert sd == pytest.approx(np.std(values, ddof=1))
    half = 2.7764451 * sd / np.sqrt(5)  # t_{0,975 ; 4}
    assert (low, high) == pytest.approx((3.0 - half, 3.0 + half))


def test_optimism_gaps_pair_seeds_and_flag_significance():
    rows = []
    for seed in range(5):
        for protocol, value in (("chrono", 0.10 + 0.001 * seed), ("random", 0.26)):
            rows.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "task": "clf",
                    "model": "xgb",
                    "strategy": "tpe",
                    "set": "test",
                    "pr_auc": value,
                    "roc_auc": 0.7,
                }
            )
    gaps = optimism_gaps(pd.DataFrame(rows)).set_index("metric")
    assert gaps.loc["pr_auc", "delta_by_seed"] == pytest.approx([0.16, 0.159, 0.158, 0.157, 0.156])
    assert gaps.loc["pr_auc", "excludes_zero"]
    assert gaps.loc["roc_auc", "delta_mean"] == 0
    assert not gaps.loc["roc_auc", "excludes_zero"]


def test_monthly_profile_skips_months_with_one_class():
    pred = pd.DataFrame(
        {
            "set": "test",
            "ordered_at": pd.to_datetime(["2018-05-02", "2018-05-20", "2018-06-03", "2018-06-10"]),
            "y_clf": [0, 1, 0, 0],  # juin n'a aucun retard
            "y_reg": [0, 3, 0, 0],
            "pred": [0.2, 0.7, 0.1, 0.3],
        }
    )
    profile = monthly_profile(pred, Task.clf)
    assert list(profile["month"]) == ["2018-05"]
    assert profile.loc[0, "prevalence"] == 0.5

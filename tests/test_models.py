"""Contrôle des pipelines : variables d'entrée, apprentissage sur l'entraînement seul, poids des classes."""

import numpy as np
import pandas as pd
import pytest

from supply_chain_resilience.config import PROCESSED_DATA_DIR
from supply_chain_resilience.models import (
    CATEGORICAL,
    NOT_FEATURES,
    NUMERIC,
    TARGET,
    Model,
    Task,
    feature_columns,
    fit_pipeline,
    make_model_pipeline,
    make_preprocessor,
    predict,
    select_features,
    task_mask,
)
from supply_chain_resilience.preprocessing import Dataset


def features_path(dataset):
    return PROCESSED_DATA_DIR / f"{dataset.value}_features.parquet"


needs_data = pytest.mark.skipif(
    not all(features_path(d).exists() for d in Dataset),
    reason="lancer make data d'abord",
)


@pytest.fixture(scope="module")
def frames():
    return {d: pd.read_parquet(features_path(d)) for d in Dataset}


# ---------------------------------------------------------------------------
# Variables d'entrée
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(Dataset))
def test_no_target_date_or_identifier_among_features(dataset):
    excluded = {c for cols in NOT_FEATURES[dataset].values() for c in cols}
    assert not excluded & set(feature_columns(dataset))
    assert not set(NUMERIC[dataset]) & set(CATEGORICAL[dataset])


@needs_data
@pytest.mark.parametrize("dataset", list(Dataset))
def test_every_column_is_either_feature_or_excluded(frames, dataset):
    excluded = [c for cols in NOT_FEATURES[dataset].values() for c in cols]
    declared = feature_columns(dataset) + excluded
    assert sorted(declared) == sorted(frames[dataset].columns)


# ---------------------------------------------------------------------------
# Le prétraitement n'apprend que sur l'entraînement
# ---------------------------------------------------------------------------


def toy(dataset, n, seed, categories):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC[dataset]})
    for c in CATEGORICAL[dataset]:
        X[c] = rng.choice(categories, size=n).astype(object)
    return X


@pytest.mark.parametrize("model", list(Model))
def test_unseen_category_and_missing_values_are_handled(model):
    dataset = Dataset.olist
    train = toy(dataset, 400, 0, ["a", "b"])
    test = toy(dataset, 50, 1, ["a", "b", "c"])  # « c » n'existe pas à l'entraînement
    test.iloc[0, 0] = np.nan
    test.loc[test.index[1], CATEGORICAL[dataset][0]] = np.nan
    prep = make_preprocessor(dataset, model).fit(train)
    out = prep.transform(test)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape[0] == len(test)
    if model is Model.linear:
        assert np.isfinite(out).all()


def test_statistics_come_from_training_rows_only():
    dataset = Dataset.olist
    train = toy(dataset, 300, 0, ["a", "b"])
    test = toy(dataset, 300, 1, ["a", "b"])
    test[NUMERIC[dataset][0]] += 1000.0  # un test très différent ne doit rien changer
    prep = make_preprocessor(dataset, Model.linear).fit(train)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(train[NUMERIC[dataset][0]].mean())
    first = prep.transform(test.iloc[:5])
    again = prep.transform(test)[:5]
    np.testing.assert_allclose(_dense(first), _dense(again))


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else a


# ---------------------------------------------------------------------------
# Poids des classes (éq. logloss) et ajustement sur données réelles
# ---------------------------------------------------------------------------


def test_balanced_weights_give_each_class_half_the_total():
    from sklearn.utils.class_weight import compute_sample_weight

    y = np.array([1] * 7 + [0] * 93)
    w = compute_sample_weight("balanced", y)
    assert w[y == 1].sum() == pytest.approx(len(y) / 2)
    assert w[y == 0].sum() == pytest.approx(len(y) / 2)


@needs_data
@pytest.mark.parametrize("dataset", list(Dataset))
@pytest.mark.parametrize("model", list(Model))
@pytest.mark.parametrize("task", list(Task))
def test_pipeline_fits_on_train_and_predicts_on_test(frames, dataset, model, task):
    df = frames[dataset]
    df = df[task_mask(df, task)]
    train = df[df["split_chrono"] == "train"].sample(1500, random_state=0)
    test = df[df["split_chrono"] == "test"].sample(500, random_state=0)
    pipe = make_model_pipeline(dataset, model, task, seed=0)
    if model is not Model.linear:
        pipe.set_params(model__n_estimators=20)
    fit_pipeline(pipe, select_features(train, dataset), train[TARGET[task]], task)
    pred = predict(pipe, select_features(test, dataset), task)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    if task is Task.clf:
        assert ((pred >= 0) & (pred <= 1)).all()


def test_huber_loss_only_where_the_delay_has_a_heavy_tail():
    from sklearn.linear_model import HuberRegressor, Ridge

    from supply_chain_resilience.models import make_estimator

    assert isinstance(make_estimator(Dataset.olist, Model.linear, Task.reg), HuberRegressor)
    assert isinstance(make_estimator(Dataset.dataco, Model.linear, Task.reg), Ridge)
    olist_xgb = make_estimator(Dataset.olist, Model.xgb, Task.reg)
    assert olist_xgb.get_params()["objective"] == "reg:pseudohubererror"
    dataco_xgb = make_estimator(Dataset.dataco, Model.xgb, Task.reg)
    assert dataco_xgb.get_params()["objective"] == "reg:squarederror"


@needs_data
def test_regression_uses_late_orders_only(frames):
    df = frames[Dataset.olist]
    rows = df[task_mask(df, Task.reg)]
    assert (rows["y_reg"] > 0).all()
    assert len(rows) == int(df["y_clf"].sum())

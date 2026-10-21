"""Phase 4, modèles : variables d'entrée et pipelines scikit-learn des trois modèles.

Chaque pipeline enchaîne le prétraitement (imputation, encodage, mise à l'échelle) et le
modèle. Ajusté sur l'entraînement d'un pli, il n'apprend ses statistiques que sur ces
données : le test ou la validation ne servent jamais à imputer ni à encoder (section 4.4).
"""

from enum import Enum

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from supply_chain_resilience.preprocessing import Dataset


class Model(str, Enum):
    linear = "linear"
    rf = "rf"
    xgb = "xgb"


class Task(str, Enum):
    clf = "clf"
    reg = "reg"


TARGET = {Task.clf: "y_clf", Task.reg: "y_reg"}

NUMERIC = {
    Dataset.dataco: [
        "Days for shipment (scheduled)",
        "Latitude",
        "Longitude",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Product Price",
        "order_month",
        "order_dayofweek",
        "order_hour",
        "hist_region_n",
        "hist_region_rate",
        "hist_shipping_mode_n",
        "hist_shipping_mode_rate",
        "hist_mode_region_n",
        "hist_mode_region_rate",
    ],
    Dataset.olist: [
        "n_items",
        "n_sellers",
        "total_price",
        "total_freight",
        "total_weight_g",
        "total_volume_cm3",
        "distance_km",
        "product_photos_qty",
        "product_name_lenght",
        "product_description_lenght",
        "payment_installments",
        "n_payments",
        "order_month",
        "order_dayofweek",
        "order_hour",
        "promised_days",
        "hist_seller_n",
        "hist_seller_rate",
        "hist_customer_state_n",
        "hist_customer_state_rate",
        "hist_route_n",
        "hist_route_rate",
    ],
}

CATEGORICAL = {
    Dataset.dataco: [
        "Type",
        "Category Name",
        "Customer Country",
        "Customer Segment",
        "Customer State",
        "Department Name",
        "Market",
        "Order Country",
        "Order Region",
        "Product Name",
        "Shipping Mode",
    ],
    Dataset.olist: [
        "customer_state",
        "seller_state",
        "product_category_name",
        "payment_type",
    ],
}

# Colonnes du fichier <jeu>_features.parquet qui ne sont pas des variables d'entrée.
NOT_FEATURES = {
    Dataset.dataco: {
        "identifiant": ["Customer Id", "Order Id", "Order Item Id", "group_id"],
        "code d'un libellé déjà retenu": ["Category Id", "Department Id", "Product Card Id"],
        "géographie fine, couverte par pays, région, État et coordonnées": [
            "Customer City",
            "Customer Zipcode",
            "Order City",
            "Order State",
        ],
        "date (remplacée par mois, jour et heure)": ["order date (DateOrders)", "delivered_at"],
        "cible ou partition": ["delta", "y_clf", "y_reg", "split_chrono"],
    },
    Dataset.olist: {
        "identifiant": ["order_id", "customer_id", "seller_id", "group_id"],
        "géographie fine, couverte par État, distance et historique": [
            "customer_zip_code_prefix",
            "customer_city",
            "seller_zip_code_prefix",
            "seller_city",
        ],
        "date (remplacée par mois, jour, heure et délai promis)": [
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "delivered_at",
        ],
        "cible ou partition": ["delta", "y_clf", "y_reg", "split_chrono"],
    },
}

# Une modalité plus rare que cette part de l'entraînement est regroupée avec les autres
# modalités rares ; une modalité absente de l'entraînement est traitée de même.
RARE_CATEGORY_SHARE = 0.005

# Forêt allégée, identique pendant le réglage et pour l'ajustement final.
RF_N_ESTIMATORS = 200
RF_MAX_SAMPLES = 0.3
RF_MAX_FEATURES = "sqrt"

# Perte de la régression (section 3.4) : Huber si le retard a une queue lourde.
# Olist : asymétrie 5,2, maximum 188 jours ; DataCo : retard borné entre 1 et 4 jours.
HUBER_LOSS = {Dataset.dataco: False, Dataset.olist: True}


def feature_columns(dataset: Dataset) -> list[str]:
    return NUMERIC[dataset] + CATEGORICAL[dataset]


def select_features(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Variables d'entrée ; les modalités passent en objet pour les encodeurs."""
    missing = [c for c in feature_columns(dataset) if c not in df.columns]
    if missing:
        raise KeyError(f"variables absentes : {missing}")
    X = df[feature_columns(dataset)].copy()
    for col in CATEGORICAL[dataset]:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X


def task_mask(df: pd.DataFrame, task: Task) -> pd.Series:
    """La régression est conditionnelle au retard : elle n'utilise que les lignes y = 1."""
    if task is Task.clf:
        return pd.Series(True, index=df.index)
    return df["y_clf"] == 1


def make_preprocessor(dataset: Dataset, model: Model) -> ColumnTransformer:
    """Modèle linéaire : indicatrices et mise à l'échelle. Arbres : codes ordinaux, NaN conservés."""
    impute_category = SimpleImputer(strategy="constant", fill_value="absent")
    if model is Model.linear:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=RARE_CATEGORY_SHARE
        )
    else:
        # La forêt aléatoire et XGBoost traitent eux-mêmes les valeurs manquantes.
        numeric = "passthrough"
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            min_frequency=RARE_CATEGORY_SHARE,
        )
    categorical = Pipeline([("impute", impute_category), ("encode", encoder)])
    return ColumnTransformer(
        [("num", numeric, NUMERIC[dataset]), ("cat", categorical, CATEGORICAL[dataset])]
    )


def make_estimator(
    dataset: Dataset, model: Model, task: Task, params: dict | None = None, seed: int = 0
):
    """Classification : entropie croisée pondérée (éq. logloss). Régression : Huber ou carré.

    La forêt aléatoire n'a pas de critère de Huber : elle garde l'erreur quadratique.
    """
    huber = task is Task.reg and HUBER_LOSS[dataset]
    if model is Model.linear:
        if task is Task.clf:
            est = LogisticRegression(max_iter=2000)
        else:
            est = HuberRegressor(max_iter=1000) if huber else Ridge()
    elif model is Model.rf:
        cls = RandomForestClassifier if task is Task.clf else RandomForestRegressor
        est = cls(
            n_estimators=RF_N_ESTIMATORS,
            max_samples=RF_MAX_SAMPLES,
            max_features=RF_MAX_FEATURES,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        common = dict(tree_method="hist", n_jobs=-1, random_state=seed)
        if task is Task.clf:
            est = XGBClassifier(objective="binary:logistic", **common)
        else:
            objective = "reg:pseudohubererror" if huber else "reg:squarederror"
            est = XGBRegressor(objective=objective, **common)
    return est.set_params(**(params or {}))


def make_model_pipeline(
    dataset: Dataset, model: Model, task: Task, params: dict | None = None, seed: int = 0
) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(dataset, model)),
            ("model", make_estimator(dataset, model, task, params, seed)),
        ]
    )


def fit_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: Task) -> Pipeline:
    """En classification, chaque classe pèse n/2 au total (poids w_c = n / (2 n_c))."""
    y = np.asarray(y, dtype=float if task is Task.reg else int)
    if task is Task.clf:
        return pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


def predict(pipe: Pipeline, X: pd.DataFrame, task: Task) -> np.ndarray:
    """Probabilité de retard en classification, retard en jours en régression."""
    if task is Task.clf:
        return pipe.predict_proba(X)[:, 1]
    return pipe.predict(X)

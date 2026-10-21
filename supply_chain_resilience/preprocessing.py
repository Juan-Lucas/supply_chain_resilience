"""Phase 3, Data Preparation : données brutes -> data/processed/<jeu>.parquet.

Opérations sans apprentissage uniquement. Tout ce qui s'apprend sur les données
(imputation, encodage) est fait dans les pipelines de la phase 4, sur l'entraînement.
"""

from enum import Enum
import json

from loguru import logger
import numpy as np
import pandas as pd
import typer

from supply_chain_resilience.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

RESULTS_DIR = REPORTS_DIR / "results"


class Dataset(str, Enum):
    dataco = "dataco"
    olist = "olist"


# ---------------------------------------------------------------------------
# Étape 2. Chargement
# ---------------------------------------------------------------------------


DATACO_FILE = "DataCoSupplyChainDataset.csv"
DATACO_DATE_FORMAT = "%m/%d/%Y %H:%M"
DATACO_DATE_COLUMNS = ["order date (DateOrders)", "shipping date (DateOrders)"]

OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}
OLIST_DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
}


def load_dataco() -> pd.DataFrame:
    """Une ligne par article commandé ; fichier en latin-1."""
    df = pd.read_csv(RAW_DATA_DIR / "dataco" / DATACO_FILE, encoding="latin-1")
    for col in DATACO_DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format=DATACO_DATE_FORMAT)
    logger.info(f"DataCo chargé : {df.shape}")
    return df


def load_olist() -> dict[str, pd.DataFrame]:
    """Les sept tables utiles d'Olist, indexées par un nom court ; fichiers en UTF-8."""
    tables = {}
    for name, file in OLIST_FILES.items():
        df = pd.read_csv(RAW_DATA_DIR / "olist" / file, encoding="utf-8")
        for col in OLIST_DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col])
        tables[name] = df
        logger.info(f"Olist {name} chargé : {df.shape}")
    return tables


# ---------------------------------------------------------------------------
# Étape 3. Cibles
# ---------------------------------------------------------------------------


def build_targets(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """delta = durée réelle - durée prévue (jours) ; y_clf = retard ; y_reg = ampleur du retard.

    Olist : jours calendaires, la date de livraison est ramenée au jour (la date estimée
    est déjà à minuit). delta reste manquant pour les commandes non livrées, que les
    filtres de l'étape 4 retirent ; les cibles sont donc des entiers nullables (Int64).

    delivered_at est l'instant où la cible devient connue (t^e de la section 3.2) : il
    sert à la partition chronologique, jamais au modèle. Pour DataCo, la date
    d'expédition redonne exactement la durée réelle (phase 2, étape 10e).
    """
    df = df.copy()
    if dataset is Dataset.dataco:
        delta = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
        df["delivered_at"] = df["shipping date (DateOrders)"]
    else:
        delivered_day = df["order_delivered_customer_date"].dt.normalize()
        delta = (delivered_day - df["order_estimated_delivery_date"]).dt.days
        df["delivered_at"] = df["order_delivered_customer_date"]

    df["delta"] = delta.astype("Int64")
    df["y_clf"] = (df["delta"] > 0).astype("Int64")
    df["y_reg"] = df["delta"].clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Étape 4. Filtres
# ---------------------------------------------------------------------------


OLIST_FIRST_YEAR = 2017
CENSORING_QUANTILE = 0.95


def olist_censoring_cutoff(orders: pd.DataFrame) -> tuple[pd.Timestamp, int, pd.Timestamp]:
    """Date à partir de laquelle les achats d'Olist sont censurés (phase 2, étape 13).

    Fin d'observation = dernière approbation de la table complète ; horizon = 95e centile
    de la durée de livraison (jours) des commandes livrées depuis 2017 ; coupure = fin
    d'observation ramenée au jour, moins l'horizon.
    """
    end_obs = orders["order_approved_at"].max()
    delivered = orders[
        orders["order_delivered_customer_date"].notna()
        & (orders["order_status"] == "delivered")
        & (orders["order_purchase_timestamp"].dt.year >= OLIST_FIRST_YEAR)
    ]
    delivery_days = (delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]).dt.days
    horizon = int(delivery_days.quantile(CENSORING_QUANTILE))
    cutoff = end_obs.normalize() - pd.Timedelta(days=horizon)
    return end_obs, horizon, cutoff


def apply_filters(df: pd.DataFrame, dataset: Dataset, summary: dict) -> pd.DataFrame:
    """Retire les lignes dont la cible n'est pas définie ou est biaisée ; note les effectifs dans summary."""
    summary["n_raw"] = len(df)

    if dataset is Dataset.dataco:
        canceled = df["Delivery Status"] == "Shipping canceled"
        summary["n_canceled"] = int(canceled.sum())
        df = df[~canceled]
    else:
        end_obs, horizon, cutoff = olist_censoring_cutoff(df)
        delivered = (df["order_status"] == "delivered") & df["order_delivered_customer_date"].notna()
        before_first_year = df["order_purchase_timestamp"].dt.year < OLIST_FIRST_YEAR
        censored = df["order_purchase_timestamp"] >= cutoff
        summary.update(
            end_of_observation=str(end_obs),
            censoring_horizon_days=horizon,
            censoring_cutoff=str(cutoff.date()),
            n_not_delivered=int((~delivered).sum()),
            n_before_first_year=int((delivered & before_first_year).sum()),
            n_censored=int((delivered & ~before_first_year & censored).sum()),
        )
        df = df[delivered & ~before_first_year & ~censored]

    df = df.astype({"delta": "int64", "y_clf": "int64", "y_reg": "int64"})
    logger.info(f"Filtres : {summary['n_raw']} -> {len(df)} lignes, prévalence {df['y_clf'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# Étape 5. Colonnes exclues
# ---------------------------------------------------------------------------


# Phase 2, étape 10 (DataCo) et étape 19 (Olist). Les clés de commande restent : elles
# servent aux partitions groupées, mais ne sont pas des variables du modèle.
DATACO_EXCLUDED = {
    "vide ou constante": ["Product Description", "Product Status", "Customer Email", "Customer Password"],
    "trop incomplète": ["Order Zipcode"],
    "donnée personnelle": ["Customer Fname", "Customer Lname", "Customer Street"],
    "fuite de la cible": [
        "Days for shipping (real)",
        "shipping date (DateOrders)",
        "Delivery Status",
        "Late_delivery_risk",
        "Order Status",
    ],
    "copie d'une autre colonne": [
        "Order Item Cardprod Id",
        "Order Customer Id",
        "Benefit per order",
        "Sales per customer",
        "Order Item Product Price",
        "Product Category Id",
        "Product Image",
    ],
}
OLIST_EXCLUDED = {
    "connue après l'achat": [
        "order_status",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ],
}


def drop_excluded_columns(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Retire les colonnes exclues ; échoue si l'une d'elles manque, pour détecter un changement de schéma."""
    excluded = DATACO_EXCLUDED if dataset is Dataset.dataco else OLIST_EXCLUDED
    columns = [col for cols in excluded.values() for col in cols]
    logger.info(f"Colonnes exclues : {len(columns)}")
    return df.drop(columns=columns)


# ---------------------------------------------------------------------------
# Étape 6. Olist : agrégation par commande
# ---------------------------------------------------------------------------


BRAZIL_LAT = (-34, 6)
BRAZIL_LNG = (-74, -34)
EARTH_RADIUS_KM = 6371


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Un point par préfixe de code postal : médiane des points situés au Brésil."""
    in_brazil = geolocation["geolocation_lat"].between(*BRAZIL_LAT) & geolocation["geolocation_lng"].between(*BRAZIL_LNG)
    return (
        geolocation[in_brazil]
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def aggregate_olist(orders: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une ligne par commande : articles, produits, vendeurs, client, paiements et distance.

    L'article principal est le plus cher de la commande : il donne la catégorie et le
    vendeur principal. La distance est le maximum, sur les articles, de la distance
    entre le vendeur et le client. Les valeurs nulles impossibles (poids de 0 g,
    0 échéance) deviennent manquantes ; leur imputation se fait en phase 4.
    """
    coords = zip_coordinates(tables["geolocation"])
    customers = tables["customers"].set_index("customer_id")

    items = tables["items"][tables["items"]["order_id"].isin(orders["order_id"])]
    items = items.merge(tables["products"], on="product_id", how="left").merge(tables["sellers"], on="seller_id", how="left")
    items["product_weight_g"] = items["product_weight_g"].replace(0, np.nan)
    items["product_volume_cm3"] = items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]

    # Distance vendeur - client, par article
    items["customer_zip_code_prefix"] = items["order_id"].map(orders.set_index("order_id")["customer_id"]).map(
        customers["customer_zip_code_prefix"]
    )
    seller_xy = coords.reindex(items["seller_zip_code_prefix"]).to_numpy()
    customer_xy = coords.reindex(items["customer_zip_code_prefix"]).to_numpy()
    items["distance_km"] = haversine_km(seller_xy[:, 0], seller_xy[:, 1], customer_xy[:, 0], customer_xy[:, 1])

    grouped = items.groupby("order_id")
    item_features = grouped.agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", lambda s: s.sum(min_count=1)),
        total_volume_cm3=("product_volume_cm3", lambda s: s.sum(min_count=1)),
        distance_km=("distance_km", "max"),
    )
    main_item = (
        items.sort_values(["price", "order_item_id"], ascending=[False, True])
        .drop_duplicates("order_id")
        .set_index("order_id")[
            [
                "product_category_name",
                "product_photos_qty",
                "product_name_lenght",
                "product_description_lenght",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ]
    )

    payments = tables["payments"][tables["payments"]["order_id"].isin(orders["order_id"])]
    main_payment = (
        payments.sort_values(["payment_value", "payment_sequential"], ascending=[False, True])
        .drop_duplicates("order_id")
        .set_index("order_id")[["payment_type", "payment_installments"]]
    )
    main_payment["payment_installments"] = main_payment["payment_installments"].replace(0, np.nan)
    main_payment["n_payments"] = payments.groupby("order_id").size()

    df = (
        orders.merge(customers[["customer_zip_code_prefix", "customer_city", "customer_state"]], left_on="customer_id", right_index=True, how="left")
        .merge(item_features, left_on="order_id", right_index=True, how="left")
        .merge(main_item, left_on="order_id", right_index=True, how="left")
        .merge(main_payment, left_on="order_id", right_index=True, how="left")
    )
    if len(df) != len(orders):
        raise ValueError(f"L'agrégation a changé le nombre de commandes : {len(orders)} -> {len(df)}")

    logger.info(f"Agrégation : {df.shape}, distance manquante pour {df['distance_km'].isna().sum()} commandes")
    return df


# ---------------------------------------------------------------------------
# Étape 7. Variables de date
# ---------------------------------------------------------------------------


ORDER_DATE = {Dataset.dataco: "order date (DateOrders)", Dataset.olist: "order_purchase_timestamp"}


def add_date_features(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Mois, jour de la semaine (0 = lundi) et heure de la commande ; pour Olist, délai promis en jours.

    Les dates restent dans le tableau sans être des variables du modèle : la date de
    commande sert aux partitions (étape 8), la date estimée d'Olist à la date annoncée
    au client (section 4.2 de l'article).
    """
    df = df.copy()
    order_date = df[ORDER_DATE[dataset]]
    df["order_month"] = order_date.dt.month
    df["order_dayofweek"] = order_date.dt.dayofweek
    df["order_hour"] = order_date.dt.hour
    if dataset is Dataset.olist:
        df["promised_days"] = (df["order_estimated_delivery_date"] - order_date.dt.normalize()).dt.days
    return df


# ---------------------------------------------------------------------------
# Étape 8. Partitions
# ---------------------------------------------------------------------------


ORDER_KEY = {Dataset.dataco: "Order Id", Dataset.olist: "order_id"}
TEST_SHARE = 0.25


def add_splits(df: pd.DataFrame, dataset: Dataset, summary: dict) -> pd.DataFrame:
    """group_id = la commande ; split_chrono = "train", "test" ou "gap".

    Contrainte de précédence (section 3.2) : avec une coupure tau*, une commande va au
    test si elle est passée à partir de tau*, et à l'entraînement si elle est livrée au
    plus tard à tau*, c'est-à-dire si sa cible est déjà connue. Les commandes passées
    avant tau* mais livrées après ("gap") ne vont dans aucun des deux ensembles.

    tau* est le quantile 1 - TEST_SHARE des dates de commande, calculé avec une date
    par commande. La partition se décide par commande, donc toutes ses lignes suivent.
    Les partitions aléatoires groupées sont faites en phase 4 à partir de group_id.
    """
    df = df.copy()
    df["group_id"] = df[ORDER_KEY[dataset]]

    orders = df.groupby("group_id").agg(ordered_at=(ORDER_DATE[dataset], "first"), delivered_at=("delivered_at", "max"))
    cut = orders["ordered_at"].quantile(1 - TEST_SHARE)
    orders["split_chrono"] = np.select(
        [orders["ordered_at"] >= cut, orders["delivered_at"] <= cut],
        ["test", "train"],
        default="gap",
    )
    df["split_chrono"] = df["group_id"].map(orders["split_chrono"])

    per_order = df.drop_duplicates("group_id")
    summary["chrono_cut"] = str(cut)
    for split in ("train", "test", "gap"):
        part = per_order[per_order["split_chrono"] == split]
        summary[f"n_orders_{split}"] = len(part)
        summary[f"prevalence_{split}"] = round(float(part["y_clf"].mean()), 4)
    logger.info(
        f"Coupure {cut} : {summary['n_orders_train']} / {summary['n_orders_test']} / {summary['n_orders_gap']} "
        f"commandes (train / test / gap), prévalence {summary['prevalence_train']} / {summary['prevalence_test']}"
    )
    return df


# ---------------------------------------------------------------------------
# Étape 9. Sorties
# ---------------------------------------------------------------------------


def save_outputs(df: pd.DataFrame, dataset: Dataset, summary: dict) -> None:
    """Écrit data/processed/<jeu>.parquet et reports/results/<jeu>_summary.json.

    Le résumé donne les chiffres de l'article (section 4.3 et tableau 2), calculés avec
    une ligne par commande : la cible est constante au sein d'une commande.
    """
    per_order = df.drop_duplicates("group_id")
    order_date = per_order[ORDER_DATE[dataset]]
    quarterly = per_order.groupby(order_date.dt.to_period("Q"))["y_clf"].mean()
    late = per_order.loc[per_order["y_clf"] == 1, "y_reg"]

    summary.update(
        n_rows=len(df),
        n_orders=len(per_order),
        period_start=str(order_date.min()),
        period_end=str(order_date.max()),
        prevalence=round(float(per_order["y_clf"].mean()), 4),
        prevalence_quarter_min=round(float(quarterly.min()), 4),
        prevalence_quarter_max=round(float(quarterly.max()), 4),
        delay_median=float(late.median()),
        delay_mean=round(float(late.mean()), 2),
        delay_max=int(late.max()),
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_DIR / f"{dataset.value}.parquet", index=False)
    with open(RESULTS_DIR / f"{dataset.value}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


@app.command()
def main(dataset: Dataset = typer.Option(..., help="Jeu de données à préparer.")):
    logger.info(f"Préparation de {dataset.value} depuis {RAW_DATA_DIR / dataset.value}")
    summary: dict = {"dataset": dataset.value}

    if dataset is Dataset.dataco:
        df = load_dataco()
        df = build_targets(df, dataset)
        df = apply_filters(df, dataset, summary)
    else:
        tables = load_olist()
        df = build_targets(tables["orders"], dataset)
        df = apply_filters(df, dataset, summary)
        df = aggregate_olist(df, tables)

    df = drop_excluded_columns(df, dataset)
    df = add_date_features(df, dataset)
    df = add_splits(df, dataset, summary)
    save_outputs(df, dataset, summary)

    logger.success(f"{dataset.value} : {len(df)} lignes écrites dans {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    app()

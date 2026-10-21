"""Phase 3, variables d'historique : data/processed/<jeu>.parquet -> <jeu>_features.parquet.

Pour une commande passée à tau_i, l'historique d'une entité (vendeur, État, région...)
ne compte que les commandes de cette entité déjà livrées à tau_i, dans une fenêtre de
HISTORY_WINDOW_DAYS jours (section 3.2 de l'article, équations hist et taux). Le calcul
n'utilise que l'information disponible à chaque instant : il est fait une seule fois et
sert aux deux protocoles d'évaluation.
"""

import json

from loguru import logger
import numpy as np
import pandas as pd
import typer

from supply_chain_resilience.config import PROCESSED_DATA_DIR
from supply_chain_resilience.preprocessing import ORDER_DATE, RESULTS_DIR, Dataset

app = typer.Typer()

HISTORY_WINDOW_DAYS = 90
HISTORY_MIN_ORDERS = 10

# Entités dont on mesure le taux de retard passé ; une liaison est un tuple de colonnes.
HISTORY_ENTITIES = {
    Dataset.olist: {
        "seller": ("seller_id",),
        "customer_state": ("customer_state",),
        "route": ("seller_state", "customer_state"),
    },
    Dataset.dataco: {
        "region": ("Order Region",),
        "shipping_mode": ("Shipping Mode",),
        "mode_region": ("Shipping Mode", "Order Region"),
    },
}


# ---------------------------------------------------------------------------
# Étape 2. Taux de retard historique
# ---------------------------------------------------------------------------


def history_rate(
    orders: pd.DataFrame,
    entity: tuple[str, ...],
    window_days: int = HISTORY_WINDOW_DAYS,
    min_orders: int = HISTORY_MIN_ORDERS,
) -> pd.DataFrame:
    """Taux de retard passé de l'entité de chaque commande, et nombre de commandes qui le fondent.

    orders contient une ligne par commande, avec les colonnes de l'entité, ordered_at
    (tau_i), delivered_at (t^e) et y_clf. L'historique H_i de la commande i regroupe les
    commandes k != i de la même entité telles que tau_i - w <= t^e_k <= tau_i. Le taux
    est manquant si |H_i| < min_orders. Renvoie les colonnes n et rate, alignées sur orders.
    """
    window = pd.Timedelta(days=window_days)
    n = np.zeros(len(orders), dtype=np.int64)
    late = np.zeros(len(orders), dtype=np.int64)

    for _, idx in orders.groupby(list(entity), dropna=False, sort=False).indices.items():
        group = orders.iloc[idx]
        by_delivery = np.argsort(group["delivered_at"].to_numpy(), kind="stable")
        delivered = group["delivered_at"].to_numpy()[by_delivery]
        cum_late = np.concatenate([[0], np.cumsum(group["y_clf"].to_numpy()[by_delivery])])

        ordered = group["ordered_at"].to_numpy()
        upper = np.searchsorted(delivered, ordered, side="right")
        lower = np.searchsorted(delivered, ordered - window.to_timedelta64(), side="left")
        n_group = upper - lower
        late_group = cum_late[upper] - cum_late[lower]

        # k != i : une commande livrée dès l'instant de sa commande ne compte pas pour elle-même
        own_delivery = group["delivered_at"].to_numpy()
        self_in = (own_delivery <= ordered) & (own_delivery >= ordered - window.to_timedelta64())
        n_group -= self_in
        late_group -= np.where(self_in, group["y_clf"].to_numpy(), 0)

        n[idx] = n_group
        late[idx] = late_group

    rate = np.where(n >= min_orders, late / np.maximum(n, 1), np.nan)
    return pd.DataFrame({"n": n, "rate": rate}, index=orders.index)


# ---------------------------------------------------------------------------
# Étape 3. Application aux entités du jeu
# ---------------------------------------------------------------------------


def add_history_features(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Ajoute hist_<entité>_n et hist_<entité>_rate pour chaque entité du jeu.

    Le calcul se fait avec une ligne par commande (group_id), puis est recopié sur
    toutes les lignes de la commande : sinon, dans DataCo, une commande de trois
    lignes compterait trois fois dans l'historique.
    """
    entities = HISTORY_ENTITIES[dataset]
    entity_columns = sorted({col for cols in entities.values() for col in cols})
    orders = (
        df.groupby("group_id", sort=False)
        .agg(
            ordered_at=(ORDER_DATE[dataset], "first"),
            delivered_at=("delivered_at", "max"),
            y_clf=("y_clf", "first"),
            **{col: (col, "first") for col in entity_columns},
        )
        .reset_index()
    )

    df = df.copy()
    for name, cols in entities.items():
        hist = history_rate(orders, cols).set_index(orders["group_id"])
        df[f"hist_{name}_n"] = df["group_id"].map(hist["n"])
        df[f"hist_{name}_rate"] = df["group_id"].map(hist["rate"])
        logger.info(
            f"{name} : taux manquant pour {hist['rate'].isna().mean():.1%} des commandes, "
            f"historique médian de {hist['n'].median():.0f} commandes"
        )
    return df


# ---------------------------------------------------------------------------
# Résumé pour l'article (section 4.4)
# ---------------------------------------------------------------------------


def summarize_history(df: pd.DataFrame, dataset: Dataset) -> dict:
    """Par entité : part de taux manquants, historique médian, prévalence par quartile du taux.

    Calculé avec une ligne par commande. Quand beaucoup de taux sont égaux, qcut peut
    former moins de quatre groupes ; top_bottom_ratio compare toujours le groupe le
    plus haut au plus bas.
    """
    orders = df.drop_duplicates("group_id")
    summary: dict = {"dataset": dataset.value, "window_days": HISTORY_WINDOW_DAYS, "min_orders": HISTORY_MIN_ORDERS}
    for name in HISTORY_ENTITIES[dataset]:
        rate = orders[f"hist_{name}_rate"]
        groups = pd.qcut(rate, 4, labels=False, duplicates="drop")
        prevalence = orders.groupby(groups)["y_clf"].mean()
        summary[name] = {
            "missing_rate_share": round(float(rate.isna().mean()), 4),
            "median_history_orders": float(orders[f"hist_{name}_n"].median()),
            "prevalence_by_quartile": [round(float(p), 4) for p in prevalence],
            "top_bottom_ratio": round(float(prevalence.iloc[-1] / prevalence.iloc[0]), 2),
        }
    return summary


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


@app.command()
def main(dataset: Dataset = typer.Option(..., help="Jeu de données à enrichir.")):
    input_path = PROCESSED_DATA_DIR / f"{dataset.value}.parquet"
    output_path = PROCESSED_DATA_DIR / f"{dataset.value}_features.parquet"
    logger.info(f"Variables d'historique de {dataset.value} depuis {input_path}")

    df = pd.read_parquet(input_path)
    df = add_history_features(df, dataset)
    df.to_parquet(output_path, index=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{dataset.value}_features_summary.json", "w", encoding="utf-8") as f:
        json.dump(summarize_history(df, dataset), f, ensure_ascii=False, indent=2)

    logger.success(f"{dataset.value} : {df.shape} écrit dans {output_path}")


if __name__ == "__main__":
    app()

"""Diagnostics des données brutes qui justifient les choix de préparation (section 4.3).

Ces chiffres ne transforment pas les données : ils documentent les décisions prises
en phase 2 (notebooks/01_eda.ipynb) et sont écrits dans reports/results/diagnostics.json.
"""

import json

from loguru import logger
import pandas as pd
import typer

from supply_chain_resilience.preprocessing import (
    OLIST_FIRST_YEAR,
    RESULTS_DIR,
    Dataset,
    apply_filters,
    build_targets,
    load_dataco,
    load_olist,
)

app = typer.Typer()

DATACO_RUPTURE = "2017-10"
LAST_WEEKS = 4


def dataco_cancellations(raw: pd.DataFrame) -> dict:
    """Lignes annulées, et celles où l'indicateur Late_delivery_risk contredit la cible reconstruite."""
    df = build_targets(raw, Dataset.dataco)
    canceled = df[df["Delivery Status"] == "Shipping canceled"]
    return {
        "n_canceled_lines": len(canceled),
        "late_delivery_risk_values": sorted(int(v) for v in canceled["Late_delivery_risk"].unique()),
        "n_indicator_disagreements": int((canceled["y_clf"] != canceled["Late_delivery_risk"]).sum()),
    }


def dataco_rupture(raw: pd.DataFrame) -> dict:
    """Produits distincts et lignes par commande, par mois, avant et après la rupture d'octobre 2017.

    Calculé sur toutes les lignes, annulations comprises, comme dans l'exploration.
    """
    month = raw["order date (DateOrders)"].dt.to_period("M")
    monthly = raw.groupby(month).agg(
        products=("Product Name", "nunique"),
        lines=("Order Id", "size"),
        orders=("Order Id", "nunique"),
    )
    monthly["lines_per_order"] = monthly["lines"] / monthly["orders"]
    before = monthly[monthly.index < pd.Period(DATACO_RUPTURE, "M")]
    after = monthly[monthly.index >= pd.Period(DATACO_RUPTURE, "M")]
    return {
        "rupture_month": DATACO_RUPTURE,
        "products_per_month_before": [int(before["products"].min()), int(before["products"].max())],
        "products_per_month_after": [int(after["products"].min()), int(after["products"].max())],
        "lines_per_order_before": [round(float(before["lines_per_order"].min()), 2), round(float(before["lines_per_order"].max()), 2)],
        "lines_per_order_after": [round(float(after["lines_per_order"].min()), 2), round(float(after["lines_per_order"].max()), 2)],
    }


def olist_shipping_limit(tables: dict[str, pd.DataFrame]) -> dict:
    """Part des articles des commandes gardées dont la date limite a la même heure que l'approbation, ou que l'achat."""
    kept = apply_filters(build_targets(tables["orders"], Dataset.olist), Dataset.olist, {})
    items = tables["items"].merge(kept[["order_id", "order_purchase_timestamp", "order_approved_at"]], on="order_id")
    limit_time = items["shipping_limit_date"].dt.time
    days_after_approval = ((items["shipping_limit_date"] - items["order_approved_at"]).dt.total_seconds() / 86400).round(4)
    return {
        "share_same_time_as_approval": round(float((limit_time == items["order_approved_at"].dt.time).mean()), 4),
        "share_same_time_as_purchase": round(float((limit_time == items["order_purchase_timestamp"].dt.time).mean()), 4),
        "most_frequent_days_after_approval": float(days_after_approval.mode().iloc[0]),
    }


def olist_censoring(tables: dict[str, pd.DataFrame]) -> dict:
    """Durée moyenne de livraison par semaine d'achat, avant troncature : chute des dernières semaines."""
    orders = tables["orders"]
    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & (orders["order_purchase_timestamp"].dt.year >= OLIST_FIRST_YEAR)
    ]
    delivery_days = (delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]).dt.days
    weekly = delivery_days.groupby(delivered["order_purchase_timestamp"].dt.to_period("W")).mean()
    last = weekly.tail(LAST_WEEKS)
    return {
        "median_weekly_mean_delivery_days": round(float(weekly.median()), 1),
        "last_weeks": {str(week): round(float(value), 2) for week, value in last.items()},
    }


@app.command()
def main():
    dataco = load_dataco()
    olist = load_olist()
    diagnostics = {
        "dataco_cancellations": dataco_cancellations(dataco),
        "dataco_rupture": dataco_rupture(dataco),
        "olist_shipping_limit": olist_shipping_limit(olist),
        "olist_censoring": olist_censoring(olist),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    logger.success(f"Diagnostics écrits dans {RESULTS_DIR / 'diagnostics.json'}")


if __name__ == "__main__":
    app()

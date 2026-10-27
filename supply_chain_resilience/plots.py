"""Figures de l'article : reports/results/<jeu>_evaluation.json -> reports/figures/*.pdf.

- profil_derive.pdf : PR-AUC mensuelle sur le test chronologique, par modèle, avec la
  prévalence du mois (valeur d'un classifieur sans pouvoir discriminant) ;
- economie_cout.pdf : économie S (éq. economie) en fonction du rapport de coûts r,
  sous les deux protocoles, par jeu et par modèle ;
- courbes_pr_roc.pdf : courbes précision-rappel et ROC de XGBoost sur le test.
Les figures montrent la stratégie de réglage TPE (Optuna), moyennée sur les graines.
Avec --lang en, les mêmes figures sont produites en anglais (suffixe _en).
"""

import json

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from supply_chain_resilience.config import FIGURES_DIR  # noqa: E402
from supply_chain_resilience.preprocessing import RESULTS_DIR, Dataset  # noqa: E402

app = typer.Typer()

STRATEGY = "tpe"
DATASET_LABEL = {Dataset.dataco: "DataCo", Dataset.olist: "Olist"}

TEXT = {
    "fr": {
        "models": {"linear": "Linéaire", "rf": "Forêt aléatoire", "xgb": "XGBoost"},
        "protocols": {"chrono": "Chronologique", "random": "Aléatoire"},
        "months": [
            "janv.",
            "févr.",
            "mars",
            "avr.",
            "mai",
            "juin",
            "juil.",
            "août",
            "sept.",
            "oct.",
            "nov.",
            "déc.",
        ],  # fmt: skip
        "month_prevalence": "Prévalence du mois",
        "drift_ylabel": "PR-AUC (test chronologique)",
        "saving": "Économie $S$",
        "cost_ratio": "Rapport de coûts $r$",
        "pr_title": "précision-rappel",
        "recall": "Rappel",
        "precision": "Précision",
        "fpr": "Taux de faux positifs",
        "tpr": "Taux de vrais positifs",
    },
    "en": {
        "models": {"linear": "Linear", "rf": "Random forest", "xgb": "XGBoost"},
        "protocols": {"chrono": "Chronological", "random": "Random"},
        "months": [
            "Jan.",
            "Feb.",
            "Mar.",
            "Apr.",
            "May",
            "June",
            "July",
            "Aug.",
            "Sept.",
            "Oct.",
            "Nov.",
            "Dec.",
        ],  # fmt: skip
        "month_prevalence": "Monthly prevalence",
        "drift_ylabel": "PR-AUC (chronological test)",
        "saving": "Saving $S$",
        "cost_ratio": "Cost ratio $r$",
        "pr_title": "precision-recall",
        "recall": "Recall",
        "precision": "Precision",
        "fpr": "False positive rate",
        "tpr": "True positive rate",
    },
}

# Langue des figures, fixée par main.
LANG = "fr"


def tr(key: str):
    return TEXT[LANG][key]


# Palette catégorielle validée (ordre fixe) et encre du graphique
SERIES = {"linear": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}
MARKERS = {"linear": "o", "rf": "s", "xgb": "^"}  # identité jamais portée par la couleur seule
PROTOCOL_STYLE = {"chrono": ("#2a78d6", "-", "o"), "random": ("#eb6834", "--", "s")}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
MODELS = ("linear", "rf", "xgb")

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
    }
)


def load(dataset: Dataset) -> dict:
    return json.loads(
        (RESULTS_DIR / f"{dataset.value}_evaluation.json").read_text(encoding="utf-8")
    )


def month_label(period: str) -> str:
    p = pd.Period(period, "M")
    return f"{tr('months')[p.month - 1]}\n{p.year}"


def drift_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, dataset in zip(axes, Dataset):
        prof = pd.DataFrame(load(dataset)["drift_profiles"])
        prof = prof[(prof["strategy"] == STRATEGY) & (prof["task"] == "clf")]
        months = sorted(prof["month"].unique())
        x = range(len(months))
        for model in MODELS:
            rows = prof[prof["model"] == model].set_index("month").loc[months]
            ax.plot(
                x,
                rows["pr_auc"],
                color=SERIES[model],
                marker=MARKERS[model],
                label=tr("models")[model],
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        prevalence = prof[prof["model"] == "xgb"].set_index("month").loc[months, "prevalence"]
        ax.plot(
            x, prevalence, color=MUTED, linestyle=":", linewidth=1.2, label=tr("month_prevalence")
        )
        ax.set_xticks(list(x), [month_label(m) for m in months])
        ax.set_title(DATASET_LABEL[dataset], loc="left", color=INK)
        ax.set_ylabel(tr("drift_ylabel"))
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left", frameon=False)
    return fig


def savings_figure() -> plt.Figure:
    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 4.0), sharex=True, sharey="row", constrained_layout=True
    )
    for row, dataset in zip(axes, Dataset):
        dec = pd.DataFrame(load(dataset)["decisions"])
        dec = dec[dec["strategy"] == STRATEGY]
        ratios = sorted(dec["cost_ratio"].unique())
        x = range(len(ratios))
        for ax, model in zip(row, MODELS):
            for protocol, (color, style, marker) in PROTOCOL_STYLE.items():
                s = dec[(dec["model"] == model) & (dec["protocol"] == protocol)]
                stats = s.groupby("cost_ratio")["saving"].agg(["mean", "std"]).loc[ratios]
                ax.errorbar(
                    x,
                    stats["mean"],
                    yerr=stats["std"],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    capsize=2,
                    elinewidth=0.8,
                    label=tr("protocols")[protocol],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=0)
            ax.set_xticks(list(x), [str(r) for r in ratios])
            ax.set_title(f"{DATASET_LABEL[dataset]}, {tr('models')[model]}", loc="left", color=INK)
            if model == "linear":
                ax.set_ylabel(tr("saving"))
            if dataset is Dataset.olist:
                ax.set_xlabel(tr("cost_ratio"))
    axes[0, 0].legend(loc="upper right", frameon=False)
    return fig


CURVE_MODEL = "xgb"


def curves_figure() -> plt.Figure:
    """Courbes précision-rappel et ROC de XGBoost sur le test, sous les deux protocoles."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    for row, dataset in zip(axes, Dataset):
        curves = pd.DataFrame(load(dataset)["test_curves"])
        curves = curves[(curves["strategy"] == STRATEGY) & (curves["model"] == CURVE_MODEL)]
        ax_pr, ax_roc = row
        for protocol, (color, style, _) in PROTOCOL_STYLE.items():
            c = curves[curves["protocol"] == protocol].iloc[0]
            label = tr("protocols")[protocol]
            ax_pr.plot(c["grid"], c["pr_precision"], color=color, linestyle=style, label=label)
            ax_pr.axhline(c["prevalence"], color=color, linestyle=":", linewidth=1.0)
            ax_roc.plot(c["grid"], c["roc_tpr"], color=color, linestyle=style, label=label)
        ax_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=0.9, zorder=0)
        ax_pr.set_title(f"{DATASET_LABEL[dataset]}, {tr('pr_title')}", loc="left", color=INK)
        ax_roc.set_title(f"{DATASET_LABEL[dataset]}, ROC", loc="left", color=INK)
        ax_pr.set_xlabel(tr("recall"))
        ax_pr.set_ylabel(tr("precision"))
        ax_roc.set_xlabel(tr("fpr"))
        ax_roc.set_ylabel(tr("tpr"))
        for ax in row:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
    axes[0, 1].legend(loc="lower right", frameon=False)
    return fig


@app.command()
def main(lang: str = typer.Option("fr", help="fr ou en")):
    global LANG
    if lang not in TEXT:
        raise typer.BadParameter("fr ou en")
    LANG = lang
    suffix = "" if lang == "fr" else "_en"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in [
        ("profil_derive", drift_figure),
        ("economie_cout", savings_figure),
        ("courbes_pr_roc", curves_figure),
    ]:
        fig = build()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{name}{suffix}.{ext}", dpi=200)
        plt.close(fig)
        logger.info(f"{FIGURES_DIR / name}{suffix}.pdf")


if __name__ == "__main__":
    app()

"""Chart generation for the pricing project (matplotlib + seaborn)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")
PALETTE = "viridis"


def _save(fig, name: str) -> str:
    path = config.CHART_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] wrote {path}")
    return str(path)


def chart_elasticity_by_segment(results_df: pd.DataFrame) -> str:
    d = results_df[results_df.scope == "customer_segment"].sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if e < config.ELASTIC_THRESHOLD else "#2980b9" for e in d.elasticity]
    ax.barh(d.group, d.elasticity, color=colors)
    ax.axvline(config.ELASTIC_THRESHOLD, ls="--", color="gray", label="Elastic threshold (-1.0)")
    for y, (e, se) in enumerate(zip(d.elasticity, d.std_error)):
        ax.errorbar(e, y, xerr=1.96 * se, fmt="none", ecolor="black", capsize=3, alpha=0.6)
    ax.set_xlabel("Own-price elasticity of demand")
    ax.set_title("Price Elasticity by Customer Segment")
    ax.legend(loc="lower right")
    return _save(fig, "elasticity_by_segment.png")


def chart_elasticity_by_region(results_df: pd.DataFrame) -> str:
    d = results_df[results_df.scope == "region"].sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if e < config.ELASTIC_THRESHOLD else "#27ae60" for e in d.elasticity]
    ax.barh(d.group, d.elasticity, color=colors)
    ax.axvline(config.ELASTIC_THRESHOLD, ls="--", color="gray")
    ax.set_xlabel("Own-price elasticity of demand")
    ax.set_title("Price Elasticity by Region (more pricing power = closer to 0)")
    return _save(fig, "elasticity_by_region.png")


def chart_revenue_margin_simulation(
    df: pd.DataFrame, overall_elasticity: float
) -> str:
    """Simulate revenue and gross profit across a sweep of uniform price changes."""
    base_price = df["net_price"].mean()
    base_demand = df["units_sold"].mean()
    base_cost = df["cost_per_unit"].mean()
    deltas = np.linspace(-0.15, 0.15, 61)
    rev, gp = [], []
    for dlt in deltas:
        p = base_price * (1 + dlt)
        q = base_demand * (p / base_price) ** overall_elasticity
        rev.append(p * q)
        gp.append((p - base_cost) * q)
    rev = np.array(rev) / rev[len(rev) // 2]
    gp = np.array(gp) / gp[len(gp) // 2]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(deltas * 100, rev, label="Relative revenue", lw=2.2)
    ax.plot(deltas * 100, gp, label="Relative gross profit", lw=2.2)
    best = deltas[int(np.argmax(gp))] * 100
    ax.axvline(best, ls="--", color="green", alpha=0.7,
               label=f"GP-optimal Δ ≈ {best:+.1f}%")
    ax.axvspan(-10, 10, color="gray", alpha=0.08, label="Allowed ±10% band")
    ax.set_xlabel("Uniform price change (%)")
    ax.set_ylabel("Index (1.0 = current)")
    ax.set_title(f"Revenue & Gross-Profit Simulation (E = {overall_elasticity:.2f})")
    ax.legend()
    return _save(fig, "revenue_margin_simulation.png")


def chart_recommended_price_changes(opt_df: pd.DataFrame) -> str:
    pivot = opt_df.pivot_table(
        index="product_category",
        columns="customer_segment",
        values="recommended_price_change_pct",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        cbar_kws={"label": "Recommended price change (%)"}, ax=ax,
    )
    ax.set_title("Recommended Price Changes by Category × Segment")
    ax.set_xlabel("Customer segment")
    ax.set_ylabel("Product category")
    return _save(fig, "recommended_price_changes.png")


def chart_segmentation_clusters(profiles: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=profiles,
        x="avg_discount_received",
        y="avg_net_price",
        hue="cluster_name",
        size="gross_margin_contribution",
        sizes=(60, 400),
        palette="tab10",
        alpha=0.85,
        ax=ax,
    )
    ax.set_xlabel("Average discount received")
    ax.set_ylabel("Average net price ($)")
    ax.set_yscale("log")
    ax.set_title("Customer Segmentation (K-Means)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save(fig, "segmentation_clusters.png")


def chart_elasticity_by_category(results_df: pd.DataFrame) -> str:
    d = results_df[results_df.scope == "product_category"].sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if e < config.ELASTIC_THRESHOLD else "#8e44ad" for e in d.elasticity]
    ax.barh(d.group, d.elasticity, color=colors)
    ax.axvline(config.ELASTIC_THRESHOLD, ls="--", color="gray", label="Elastic threshold (-1.0)")
    for y, (e, se) in enumerate(zip(d.elasticity, d.std_error)):
        ax.errorbar(e, y, xerr=1.96 * se, fmt="none", ecolor="black", capsize=3, alpha=0.6)
    for y, e in enumerate(d.elasticity):
        ax.text(e - 0.05, y, f"{e:.2f}", va="center", ha="right", fontsize=9)
    ax.set_xlabel("Own-price elasticity of demand")
    ax.set_title("Price Elasticity by Product Category")
    ax.legend(loc="lower left")
    return _save(fig, "elasticity_by_category.png")


def chart_sales_trend(df: pd.DataFrame) -> str:
    ts = df.groupby("date").agg(revenue=("revenue", "sum"),
                                units=("units_sold", "sum")).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ts.date, ts.revenue, color="#16324f", lw=2.2, label="Revenue")
    ax1.set_ylabel("Monthly revenue ($)", color="#16324f")
    ax1.tick_params(axis="y", labelcolor="#16324f")
    ax2 = ax1.twinx()
    ax2.plot(ts.date, ts.units, color="#1f8a8a", lw=1.6, ls="--", label="Units")
    ax2.set_ylabel("Monthly units", color="#1f8a8a")
    ax2.tick_params(axis="y", labelcolor="#1f8a8a")
    ax1.set_title("Sales Trend Over the 3-Year Panel (units & revenue, with seasonality)")
    ax1.set_xlabel("Date")
    fig.autofmt_xdate()
    return _save(fig, "sales_trend.png")


def chart_recommendation_mix(opt_df: pd.DataFrame) -> str:
    g = opt_df.groupby("recommendation_type").agg(
        count=("recommendation_type", "size"),
        gp=("expected_gross_profit_impact", "sum"),
    ).reset_index()
    g["short"] = g["recommendation_type"].str.split(":").str[0]
    g = g.sort_values("gp")
    fig, (axc, axg) = plt.subplots(1, 2, figsize=(12, 5))
    axc.barh(g.short, g["count"], color="#2980b9")
    axc.set_title("Recommendations by type (count)")
    for y, v in enumerate(g["count"]):
        axc.text(v, y, f" {int(v)}", va="center", fontsize=9)
    axg.barh(g.short, g["gp"], color="#27ae60")
    axg.set_title("Expected gross-profit impact by type ($)")
    for y, v in enumerate(g["gp"]):
        axg.text(v, y, f" ${v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, "recommendation_mix.png")


def chart_top_opportunities(opt_df: pd.DataFrame, n: int = 10) -> str:
    d = opt_df.nlargest(n, "expected_gross_profit_impact").iloc[::-1]
    labels = [f"{r.product_category[:14]} | {r.customer_segment[:10]} | {r.region}"
              for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, d.expected_gross_profit_impact, color="#16324f")
    for b, v, dlt in zip(bars, d.expected_gross_profit_impact, d.recommended_price_change_pct):
        ax.text(v, b.get_y() + b.get_height() / 2, f" ${v:,.0f} ({dlt:+.1f}%)",
                va="center", fontsize=8.5)
    ax.set_xlabel("Expected gross-profit impact over horizon ($)")
    ax.set_title(f"Top {n} Gross-Profit Opportunities (cell-level)")
    fig.tight_layout()
    return _save(fig, "top_opportunities.png")


def chart_elasticity_by_cluster(cluster_elast: pd.DataFrame) -> str:
    if cluster_elast is None or cluster_elast.empty:
        return ""
    d = cluster_elast.sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if e < config.ELASTIC_THRESHOLD else "#e67e22" for e in d.elasticity]
    ax.barh(d.cluster_name, d.elasticity, color=colors)
    ax.axvline(config.ELASTIC_THRESHOLD, ls="--", color="gray")
    for y, e in enumerate(d.elasticity):
        ax.text(e - 0.03, y, f"{e:.2f}", va="center", ha="right", fontsize=9)
    ax.set_xlabel("Own-price elasticity of demand")
    ax.set_title("Price Elasticity by Customer Cluster (K-Means)")
    return _save(fig, "elasticity_by_cluster.png")


def generate_all_charts(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    profiles: pd.DataFrame,
    overall_elasticity: float,
    cluster_elast: pd.DataFrame | None = None,
) -> dict[str, str]:
    charts = {
        "segment": chart_elasticity_by_segment(results_df),
        "region": chart_elasticity_by_region(results_df),
        "category": chart_elasticity_by_category(results_df),
        "simulation": chart_revenue_margin_simulation(df, overall_elasticity),
        "price_changes": chart_recommended_price_changes(opt_df),
        "clusters": chart_segmentation_clusters(profiles),
        "sales_trend": chart_sales_trend(df),
        "rec_mix": chart_recommendation_mix(opt_df),
        "top_opps": chart_top_opportunities(opt_df),
    }
    cl = chart_elasticity_by_cluster(cluster_elast)
    if cl:
        charts["cluster_elast"] = cl
    return charts

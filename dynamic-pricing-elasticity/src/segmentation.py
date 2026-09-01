"""K-Means customer segmentation over behavioural pricing features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config


def build_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the panel to one row per (customer_segment, region) customer
    cell with the behavioural features used for clustering."""
    grp = df.groupby(["customer_segment", "region"])
    profiles = grp.agg(
        avg_order_value=("revenue", "mean"),
        avg_units_sold=("units_sold", "mean"),
        avg_discount_received=("discount_pct", "mean"),
        order_frequency=("order_frequency", "mean"),
        gross_margin_contribution=("gross_profit", "sum"),
        avg_net_price=("net_price", "mean"),
    ).reset_index()
    return profiles


def _name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """Map raw cluster ids to business-friendly names using simple heuristics
    on discount level, margin contribution, and price level."""
    names: dict[int, str] = {}
    summary = profiles.groupby("cluster").agg(
        disc=("avg_discount_received", "mean"),
        margin=("gross_margin_contribution", "mean"),
        price=("avg_net_price", "mean"),
        units=("avg_units_sold", "mean"),
    )
    # Rank-based labelling.
    disc_rank = summary["disc"].rank(ascending=False)
    margin_rank = summary["margin"].rank(ascending=False)
    price_rank = summary["price"].rank(ascending=False)

    for cid in summary.index:
        if margin_rank[cid] == 1:
            names[cid] = "Stable High-Margin Customers"
        elif disc_rank[cid] == 1:
            names[cid] = "Promotion-Responsive Buyers"
        elif price_rank[cid] == 1:
            names[cid] = "Premium / Service-Oriented Buyers"
        elif price_rank[cid] >= len(summary) - 1:
            names[cid] = "Price-Sensitive Buyers"
        else:
            names[cid] = "Low-Volume Opportunistic Buyers"
    # Ensure uniqueness; fall back to canonical names if collisions occur.
    seen: set[str] = set()
    canon = list(config.BUSINESS_CLUSTER_NAMES)
    for cid in summary.index:
        if names[cid] in seen:
            for alt in canon:
                if alt not in seen:
                    names[cid] = alt
                    break
        seen.add(names[cid])
    return names


def run_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    profiles = build_customer_profiles(df)
    X = profiles[config.CLUSTER_FEATURES].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_SEED, n_init=10)
    profiles["cluster"] = km.fit_predict(Xs)
    profiles["cluster_name"] = profiles["cluster"].map(_name_clusters(profiles))

    print(
        f"[segmentation] {config.N_CLUSTERS} clusters over "
        f"{len(profiles)} customer cells: "
        f"{sorted(profiles['cluster_name'].unique())}"
    )
    return profiles


def attach_clusters(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Map cluster labels back onto the transaction-level panel."""
    key = profiles[["customer_segment", "region", "cluster", "cluster_name"]]
    return df.merge(key, on=["customer_segment", "region"], how="left")

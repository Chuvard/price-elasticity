"""Constrained price optimization using estimated elasticities.

For each product-category x segment x region cell we search for the price
change in [-MAX_PRICE_CHANGE, +MAX_PRICE_CHANGE] that maximises the chosen
objective (gross profit or revenue), subject to a margin floor. Demand response
follows the constant-elasticity form:

    new_demand = current_demand * (new_price / current_price) ** elasticity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import config


def _elasticity_lookup(
    results_df: pd.DataFrame, seg_region_df: pd.DataFrame, overall: float
) -> tuple[dict, dict, dict, dict]:
    """Build fast lookup dicts for elasticity by various scopes."""
    by_segment = {
        r.group: (r.elasticity, r.p_value)
        for r in results_df[results_df.scope == "customer_segment"].itertuples()
    }
    by_region = {
        r.group: (r.elasticity, r.p_value)
        for r in results_df[results_df.scope == "region"].itertuples()
    }
    by_category = {
        r.group: (r.elasticity, r.p_value)
        for r in results_df[results_df.scope == "product_category"].itertuples()
    }
    by_seg_region = {}
    if not seg_region_df.empty:
        by_seg_region = {
            (r.customer_segment, r.region): (r.elasticity, r.p_value)
            for r in seg_region_df.itertuples()
        }
    return by_segment, by_region, by_category, by_seg_region


def _best_elasticity(seg, reg, cat, lookups, overall):
    """Prefer the most specific reliable estimate available."""
    by_segment, by_region, by_category, by_seg_region = lookups
    # most specific: segment x region
    if (seg, reg) in by_seg_region:
        e, p = by_seg_region[(seg, reg)]
        if e < -0.05:
            return e, p, "segment_region"
    # blend category & segment & region signals
    candidates = []
    for src, key, d in (
        ("category", cat, by_category),
        ("segment", seg, by_segment),
        ("region", reg, by_region),
    ):
        if key in d and d[key][0] < -0.05:
            candidates.append(d[key])
    if candidates:
        e = float(np.mean([c[0] for c in candidates]))
        p = float(np.max([c[1] for c in candidates]))
        return e, p, "blended"
    return overall, 0.0, "overall"


def _optimize_cell(current_price, cost, current_demand, elasticity, objective):
    """Return optimal price change fraction within constraints."""

    def neg_objective(delta: float) -> float:
        new_price = current_price * (1 + delta)
        margin = (new_price - cost) / new_price if new_price > 0 else -1
        if margin < config.MIN_MARGIN_FLOOR:
            return 1e18  # infeasible
        new_demand = current_demand * (new_price / current_price) ** elasticity
        if objective == "revenue":
            val = new_price * new_demand
        else:
            val = (new_price - cost) * new_demand
        return -val

    res = minimize_scalar(
        neg_objective,
        bounds=(-config.MAX_PRICE_CHANGE, config.MAX_PRICE_CHANGE),
        method="bounded",
    )
    return float(res.x)


def _classify(delta, elasticity, p_value, avg_discount) -> str:
    if p_value >= config.ELASTICITY_CONFIDENCE_PMAX or elasticity > -0.05:
        return "Review manually: insufficient confidence or unstable estimate"
    if delta > 0.01 and elasticity > config.ELASTIC_THRESHOLD:
        return "Increase price: low elasticity / pricing power"
    if delta > 0.01 and avg_discount > 0.10:
        return "Reduce discount: discount not generating enough volume"
    if delta < -0.01 and elasticity < config.ELASTIC_THRESHOLD:
        return "Target promotion: elastic segment"
    return "Hold price: no clear improvement"


def run_optimization(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    seg_region_df: pd.DataFrame,
    overall_elasticity: float,
) -> pd.DataFrame:
    lookups = _elasticity_lookup(results_df, seg_region_df, overall_elasticity)

    cells = df.groupby(
        ["product_category", "customer_segment", "region"]
    ).agg(
        current_price=("net_price", "mean"),
        cost_per_unit=("cost_per_unit", "mean"),
        current_demand=("units_sold", "mean"),
        avg_discount=("discount_pct", "mean"),
        total_units=("units_sold", "sum"),
    ).reset_index()

    records: list[dict] = []
    for c in cells.itertuples():
        elasticity, pval, src = _best_elasticity(
            c.customer_segment, c.region, c.product_category, lookups, overall_elasticity
        )
        delta = _optimize_cell(
            c.current_price, c.cost_per_unit, c.current_demand,
            elasticity, config.OPTIMIZATION_OBJECTIVE,
        )
        new_price = c.current_price * (1 + delta)
        new_demand = c.current_demand * (new_price / c.current_price) ** elasticity

        cur_rev = c.current_price * c.current_demand
        new_rev = new_price * new_demand
        cur_gp = (c.current_price - c.cost_per_unit) * c.current_demand
        new_gp = (new_price - c.cost_per_unit) * new_demand

        scale = c.total_units / max(c.current_demand, 1)  # periods of data

        rec_type = _classify(delta, elasticity, pval, c.avg_discount)
        records.append(
            {
                "product_category": c.product_category,
                "customer_segment": c.customer_segment,
                "region": c.region,
                "elasticity_used": round(elasticity, 4),
                "elasticity_source": src,
                "p_value": round(pval, 4),
                "current_price": round(c.current_price, 2),
                "recommended_price": round(new_price, 2),
                "recommended_price_change_pct": round(delta * 100, 2),
                "expected_demand_change_pct": round(
                    (new_demand / c.current_demand - 1) * 100, 2
                ),
                "current_margin_pct": round(
                    (c.current_price - c.cost_per_unit) / c.current_price * 100, 2
                ),
                "baseline_revenue": round(cur_rev * scale, 2),
                "baseline_gross_profit": round(cur_gp * scale, 2),
                "expected_revenue_impact": round((new_rev - cur_rev) * scale, 2),
                "expected_gross_profit_impact": round((new_gp - cur_gp) * scale, 2),
                "recommendation_type": rec_type,
            }
        )

    out = pd.DataFrame(records).sort_values(
        "expected_gross_profit_impact", ascending=False
    ).reset_index(drop=True)
    print(
        f"[optimization] {len(out)} cells optimised; "
        f"total expected GP impact = ${out['expected_gross_profit_impact'].sum():,.0f}"
    )
    return out

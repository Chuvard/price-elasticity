"""Turn model + optimization outputs into client-facing recommendations.

Produces:
  * pricing_recommendations.csv  - actionable, prioritised recommendation list
  * business_impact_recommendations.md - narrative tied to ACTUAL output numbers
"""
from __future__ import annotations

import pandas as pd

from . import config


def build_pricing_recommendations(opt_df: pd.DataFrame) -> pd.DataFrame:
    """Prioritised, deduplicated, client-facing recommendation table."""
    recs = opt_df.copy()
    recs["abs_gp_impact"] = recs["expected_gross_profit_impact"].abs()
    recs = recs.sort_values("expected_gross_profit_impact", ascending=False)
    cols = [
        "product_category", "customer_segment", "region",
        "recommendation_type", "elasticity_used",
        "current_price", "recommended_price", "recommended_price_change_pct",
        "expected_demand_change_pct", "expected_revenue_impact",
        "expected_gross_profit_impact",
    ]
    recs = recs[cols].reset_index(drop=True)
    recs.to_csv(config.PRICING_RECS_PATH, index=False)
    print(f"[recommendations] wrote {config.PRICING_RECS_PATH}")
    return recs


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def build_business_markdown(
    results_df: pd.DataFrame,
    seg_region_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    overall: dict,
) -> str:
    seg = results_df[results_df.scope == "customer_segment"].copy()
    reg = results_df[results_df.scope == "region"].copy()
    cat = results_df[results_df.scope == "product_category"].copy()

    # Lowest elasticity = strongest pricing power (closest to 0 / least negative)
    least_elastic_seg = seg.loc[seg.elasticity.idxmax()]
    most_elastic_seg = seg.loc[seg.elasticity.idxmin()]
    strongest_region = reg.loc[reg.elasticity.idxmax()]
    weakest_region = reg.loc[reg.elasticity.idxmin()]
    least_elastic_cat = cat.loc[cat.elasticity.idxmax()]
    most_elastic_cat = cat.loc[cat.elasticity.idxmin()]

    total_rev = opt_df["expected_revenue_impact"].sum()
    total_gp = opt_df["expected_gross_profit_impact"].sum()
    base_rev = opt_df["baseline_revenue"].sum()
    base_gp = opt_df["baseline_gross_profit"].sum()
    rev_uplift_pct = (total_rev / base_rev * 100) if base_rev else 0.0
    gp_uplift_pct = (total_gp / base_gp * 100) if base_gp else 0.0

    increases = opt_df[opt_df.recommendation_type.str.startswith("Increase")]
    discounts = opt_df[opt_df.recommendation_type.str.startswith("Reduce")]
    promos = opt_df[opt_df.recommendation_type.str.startswith("Target")]
    review = opt_df[opt_df.recommendation_type.str.startswith("Review")]

    top_gp = opt_df.nlargest(5, "expected_gross_profit_impact")
    leak = discounts.nlargest(5, "expected_gross_profit_impact")
    promo_targets = promos.nsmallest(5, "expected_demand_change_pct")

    lines: list[str] = []
    A = lines.append
    A(f"# Business Impact & Pricing Recommendations — {config.SYNTHETIC_COMPANY_NAME}")
    A("")
    A(f"_Generated from model outputs. {config.DISCLAIMER}_")
    A("")
    A("## Executive Summary")
    A("")
    A(
        f"Across the analysed portfolio, the estimated **overall own-price "
        f"elasticity of demand is {overall['elasticity']:.2f}** "
        f"(robust SE {overall['std_error']:.2f}, R² {overall['r_squared']:.2f}, "
        f"n = {overall['n_obs']:,}). A value of {overall['elasticity']:.2f} means a "
        f"1% increase in net price is associated with roughly a "
        f"{abs(overall['elasticity']):.2f}% change in demand, holding seasonality, "
        f"marketing, competitor pricing, region, product mix and customer segment "
        f"constant via fixed effects."
    )
    A("")
    rev_word = "uplift" if total_rev >= 0 else "trade-off"
    A(
        f"Constrained optimization (objective: **{config.OPTIMIZATION_OBJECTIVE.replace('_', ' ')}**, "
        f"price moves capped at ±{int(config.MAX_PRICE_CHANGE*100)}%, "
        f"margin floor {int(config.MIN_MARGIN_FLOOR*100)}%) identifies "
        f"**{_fmt_money(total_gp)} of incremental gross profit "
        f"({gp_uplift_pct:+.1f}%)** over the analysed horizon, against a baseline of "
        f"{_fmt_money(base_gp)} gross profit. This comes with a modest revenue "
        f"{rev_word} of **{rev_uplift_pct:+.1f}%** "
        f"({_fmt_money(total_rev)} on a {_fmt_money(base_rev)} base) — the expected "
        f"signature of trading a little volume for materially higher margin, all "
        f"within governance constraints."
    )
    A("")
    A("## Where the Pricing Power Is")
    A("")
    A(
        f"- **Lowest elasticity (strongest pricing power) segment:** "
        f"`{least_elastic_seg.group}` at **{least_elastic_seg.elasticity:.2f}** — "
        f"these buyers tolerate price increases with little volume loss."
    )
    A(
        f"- **Highest elasticity (most price-sensitive) segment:** "
        f"`{most_elastic_seg.group}` at **{most_elastic_seg.elasticity:.2f}** — "
        f"protect volume here; use targeted promotions rather than blanket increases."
    )
    A(
        f"- **Region with strongest pricing power:** `{strongest_region.group}` "
        f"({strongest_region.elasticity:.2f}); most sensitive region: "
        f"`{weakest_region.group}` ({weakest_region.elasticity:.2f})."
    )
    A(
        f"- **Category with most pricing power:** `{least_elastic_cat.group}` "
        f"({least_elastic_cat.elasticity:.2f}); most elastic category: "
        f"`{most_elastic_cat.group}` ({most_elastic_cat.elasticity:.2f})."
    )
    A("")
    A("## Recommended Actions")
    A("")
    A(
        f"The optimizer produced **{len(increases)} price-increase**, "
        f"**{len(discounts)} discount-reduction**, **{len(promos)} targeted-promotion**, "
        f"and **{len(review)} manual-review** recommendations across "
        f"{len(opt_df)} product-category × segment × region cells."
    )
    A("")
    A("### Top 5 gross-profit opportunities")
    A("")
    A("| Category | Segment | Region | Action | Price Δ% | GP impact |")
    A("|---|---|---|---|---:|---:|")
    for r in top_gp.itertuples():
        A(
            f"| {r.product_category} | {r.customer_segment} | {r.region} | "
            f"{r.recommendation_type.split(':')[0]} | "
            f"{r.recommended_price_change_pct:+.1f}% | "
            f"{_fmt_money(r.expected_gross_profit_impact)} |"
        )
    A("")
    A("### Where discounts should be reduced (margin leakage)")
    A("")
    if len(leak):
        for r in leak.itertuples():
            A(
                f"- **{r.product_category} / {r.customer_segment} / {r.region}**: "
                f"raise net price {r.recommended_price_change_pct:+.1f}% "
                f"(elasticity {r.elasticity_used:.2f}); "
                f"+{_fmt_money(r.expected_gross_profit_impact)} GP."
            )
    else:
        A("- No material discount-reduction cells flagged at current thresholds.")
    A("")
    A("### Where targeted promotions should be used (elastic cells)")
    A("")
    if len(promo_targets):
        for r in promo_targets.itertuples():
            A(
                f"- **{r.product_category} / {r.customer_segment} / {r.region}**: "
                f"elasticity {r.elasticity_used:.2f}; a targeted price reduction of "
                f"{r.recommended_price_change_pct:+.1f}% is expected to lift demand "
                f"{r.expected_demand_change_pct:+.1f}%."
            )
    else:
        A("- No strongly elastic promotion cells flagged at current thresholds.")
    A("")
    A("## Practical Next Steps")
    A("")
    A("**Finance:** lock margin floors per category, fund the price-increase cells "
      "above first (lowest execution risk, highest GP), and require model sign-off "
      "for any change beyond ±10%.")
    A("")
    A("**Marketing:** redirect promotional budget away from low-elasticity segments "
      f"(e.g. `{least_elastic_seg.group}`) toward the elastic cells listed above, "
      "where discounts actually buy incremental volume.")
    A("")
    A("**Sales:** use the recommended price bands as guardrails in quoting tools; "
      "escalate manual-review cells rather than discounting on instinct.")
    A("")
    A("## How a Client Can Use This Approach")
    A("")
    A("- **Replace blanket discounting with targeted discounting** driven by "
      "measured segment elasticity instead of habit.")
    A("- **Protect margin in low-elasticity segments** by holding or raising price "
      "where volume is insensitive.")
    A("- **Identify product categories with pricing power** and prioritise them for "
      "list-price review.")
    A("- **Simulate price changes before implementation** using the demand-response "
      "equation, avoiding costly live experiments.")
    A("- **Align Finance, Marketing, and Sales** around one shared, data-driven set "
      "of pricing rules.")
    A("- **Build a recurring pricing governance process** — monthly re-estimation, "
      "constraint review, and a single approval workflow.")
    A("")
    return "\n".join(lines)


def write_business_markdown(
    results_df, seg_region_df, opt_df, overall
) -> str:
    md = build_business_markdown(results_df, seg_region_df, opt_df, overall)
    config.BUSINESS_MD_PATH.write_text(md, encoding="utf-8")
    print(f"[recommendations] wrote {config.BUSINESS_MD_PATH}")
    return md


def write_segment_elasticity_summary(
    results_df: pd.DataFrame, cluster_profiles: pd.DataFrame
) -> pd.DataFrame:
    """Summary CSV combining segment elasticity with cluster labels."""
    seg = results_df[results_df.scope == "customer_segment"][
        ["group", "elasticity", "std_error", "p_value", "n_obs", "is_elastic"]
    ].rename(columns={"group": "customer_segment"})
    cluster_map = (
        cluster_profiles.groupby("customer_segment")["cluster_name"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    summary = seg.merge(cluster_map, on="customer_segment", how="left")
    summary.to_csv(config.SEGMENT_ELASTICITY_PATH, index=False)
    print(f"[recommendations] wrote {config.SEGMENT_ELASTICITY_PATH}")
    return summary

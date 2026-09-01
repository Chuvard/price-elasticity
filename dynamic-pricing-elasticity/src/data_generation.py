"""Generate a synthetic historical panel dataset for pricing analytics.

The generator embeds realistic economic relationships (downward-sloping demand,
heterogeneous elasticity by segment/region/category, marketing lift, competitor
substitution, seasonality, and macro effects) plus random noise, so that the
downstream econometric model can *recover* the latent elasticities.

No real company data is used. See config.DISCLAIMER.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _seasonality_multiplier(month: int, category: str) -> float:
    """Demand seasonality: water treatment peaks in warm months; service
    contracts renew around year start."""
    base = 1.0 + 0.18 * np.sin((month - 4) / 12.0 * 2 * np.pi)
    if category == "Service Contracts":
        base = 1.0 + 0.22 * np.cos((month - 1) / 12.0 * 2 * np.pi)
    return float(base)


def generate_dataset() -> pd.DataFrame:
    """Build and return the full synthetic panel as a DataFrame."""
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Build the product master.
    products: list[dict] = []
    for category in config.PRODUCT_CATEGORIES:
        for i in range(config.N_PRODUCTS_PER_CATEGORY):
            pid = f"{category[:3].upper()}-{i + 1:03d}"
            list_price = config.CATEGORY_LIST_PRICE[category] * rng.uniform(0.85, 1.15)
            cost_ratio = config.CATEGORY_COST_RATIO[category] * rng.uniform(0.95, 1.05)
            products.append(
                {
                    "product_id": pid,
                    "product_category": category,
                    "base_list_price": list_price,
                    "cost_per_unit": list_price * cost_ratio,
                }
            )
    product_df = pd.DataFrame(products)

    dates = pd.date_range(
        start=f"{config.START_YEAR}-01-01", periods=config.N_PERIODS, freq="MS"
    )

    rows: list[dict] = []
    for region in config.REGIONS:
        region_power = config.REGION_PRICING_POWER[region]
        region_demand_scale = {
            "North": 1.05, "South": 1.20, "Midwest": 1.00,
            "West": 1.15, "Northeast": 0.95,
        }[region]
        for segment in config.CUSTOMER_SEGMENTS:
            seg_power = config.SEGMENT_PRICING_POWER[segment]
            seg_demand_scale = {
                "Homeowners": 1.30, "Small Business": 1.00, "Municipal": 0.70,
                "Industrial Accounts": 0.55, "Distributors": 1.40,
            }[segment]
            for _, prod in product_df.iterrows():
                category = prod["product_category"]
                base_elasticity = config.CATEGORY_BASE_ELASTICITY[category]
                # Effective latent elasticity for this cell.
                elasticity = base_elasticity * region_power * seg_power
                base_units = (
                    config.CATEGORY_BASE_UNITS[category]
                    * region_demand_scale
                    * seg_demand_scale
                    * rng.uniform(0.85, 1.15)
                    * config.DEMAND_SCALE
                )
                # Competitor reference index baseline for this product line.
                comp_base = rng.uniform(0.95, 1.05)

                for period_idx, date in enumerate(dates):
                    month = date.month
                    year = date.year

                    # --- Pricing levers ---
                    # Discount depth: more aggressive for elastic segments/regions.
                    disc_mean = 0.06 + 0.05 * max(0.0, (seg_power - 0.8))
                    discount_pct = float(np.clip(rng.normal(disc_mean, 0.04), 0.0, 0.35))
                    # Exogenous list-price variation (inflation trend + independent
                    # monthly shocks). This independent variation is what identifies
                    # the price elasticity separately from the discount lever.
                    price_shock = float(np.exp(rng.normal(0.0, 0.07)))
                    list_price = (
                        prod["base_list_price"]
                        * (1 + 0.02 * (year - config.START_YEAR))
                        * price_shock
                    )
                    net_price = list_price * (1 - discount_pct)

                    # --- Marketing ---
                    marketing_flag = int(rng.random() < 0.25)
                    marketing_spend = (
                        marketing_flag * rng.uniform(2000, 15000)
                        * seg_demand_scale
                    )

                    # --- Exogenous indices ---
                    competitor_price_index = float(
                        comp_base + 0.05 * np.sin(period_idx / 6.0) + rng.normal(0, 0.03)
                    )
                    category_market_index = float(
                        1.0 + 0.04 * np.cos(period_idx / 9.0) + rng.normal(0, 0.02)
                    )
                    inflation_index = float(
                        1.0 + 0.025 * (period_idx / 12.0) + rng.normal(0, 0.005)
                    )
                    season = _seasonality_multiplier(month, category)

                    # --- Demand model (log-linear, then exponentiated) ---
                    price_ratio = net_price / prod["base_list_price"]
                    log_demand = (
                        np.log(base_units)
                        + elasticity * np.log(price_ratio)
                        + 0.35 * discount_pct          # modest extra discount lift
                        + 0.16 * marketing_flag
                        + 0.45 * np.log(max(competitor_price_index, 0.5))  # substitution
                        + np.log(season)
                        + 0.10 * np.log(category_market_index)
                        - 0.30 * np.log(inflation_index)
                        + rng.normal(0, 0.12)          # random noise
                    )
                    units_sold = int(round(np.exp(log_demand)))
                    if units_sold < 1:
                        # No sale in this cell-period: realistic sparse panel.
                        continue

                    revenue = net_price * units_sold
                    gross_profit = (net_price - prod["cost_per_unit"]) * units_sold
                    gross_margin_pct = (
                        (net_price - prod["cost_per_unit"]) / net_price
                        if net_price > 0 else 0.0
                    )
                    order_frequency = float(
                        np.clip(rng.normal(seg_demand_scale * 4, 1.2), 0.5, 12)
                    )

                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "year": year,
                            "month": month,
                            "period": period_idx + 1,
                            "region": region,
                            "customer_segment": segment,
                            "product_category": category,
                            "product_id": prod["product_id"],
                            "units_sold": units_sold,
                            "list_price": round(list_price, 2),
                            "net_price": round(net_price, 2),
                            "discount_pct": round(discount_pct, 4),
                            "cost_per_unit": round(prod["cost_per_unit"], 2),
                            "gross_margin_pct": round(gross_margin_pct, 4),
                            "revenue": round(revenue, 2),
                            "gross_profit": round(gross_profit, 2),
                            "marketing_campaign_flag": marketing_flag,
                            "marketing_spend": round(marketing_spend, 2),
                            "competitor_price_index": round(competitor_price_index, 4),
                            "category_market_index": round(category_market_index, 4),
                            "inflation_index": round(inflation_index, 4),
                            "season_index": round(season, 4),
                            "order_frequency": round(order_frequency, 3),
                            # latent truth retained for validation / teaching only
                            "_true_elasticity": round(elasticity, 4),
                        }
                    )

    df = pd.DataFrame(rows)
    return df


def generate_and_save() -> pd.DataFrame:
    config.ensure_dirs()
    df = generate_dataset()
    df.to_csv(config.DATA_PATH, index=False)
    print(f"[data_generation] Wrote {len(df):,} rows -> {config.DATA_PATH}")
    return df

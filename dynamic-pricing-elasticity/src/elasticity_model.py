"""Log-log demand regression for price elasticity estimation.

Core model (OLS with fixed effects and heteroskedasticity-robust SEs):

    log(units) = b0 + b1*log(net_price) + b2*discount_pct
                 + b3*marketing_flag + b4*log(competitor_index)
                 + b5*log(category_index) + b6*log(inflation)
                 + region FE + product_category FE + customer_segment FE
                 + month FE + e

b1 is the own-price elasticity of demand. We also re-fit the model on
subsets to obtain elasticity by segment, region, and category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config

BASE_FORMULA = (
    "log_units ~ log_net_price + discount_pct + marketing_campaign_flag "
    "+ log_competitor_index + log_category_index + log_inflation "
    "+ C(region) + C(product_category) + C(product_id) "
    "+ C(customer_segment) + C(month)"
)


def _fit(df: pd.DataFrame, formula: str = BASE_FORMULA):
    """Fit OLS with HC3 robust standard errors."""
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    return model


def estimate_overall(df: pd.DataFrame) -> dict:
    model = _fit(df)
    coef = model.params["log_net_price"]
    se = model.bse["log_net_price"]
    pval = model.pvalues["log_net_price"]
    ci_low, ci_high = model.conf_int().loc["log_net_price"].tolist()
    return {
        "scope": "overall",
        "group": "ALL",
        "elasticity": float(coef),
        "std_error": float(se),
        "p_value": float(pval),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
    }


def _estimate_by(df: pd.DataFrame, by: str, drop_fe: str) -> list[dict]:
    """Re-fit the model within each level of `by`, dropping the corresponding
    fixed effect from the formula to avoid collinearity."""
    formula = BASE_FORMULA.replace(f" + C({drop_fe})", "")
    results: list[dict] = []
    for level, sub in df.groupby(by):
        if len(sub) < 60 or sub["log_net_price"].nunique() < 10:
            continue
        try:
            model = _fit(sub, formula)
            if "log_net_price" not in model.params:
                continue
            results.append(
                {
                    "scope": by,
                    "group": str(level),
                    "elasticity": float(model.params["log_net_price"]),
                    "std_error": float(model.bse["log_net_price"]),
                    "p_value": float(model.pvalues["log_net_price"]),
                    "ci_low": float(model.conf_int().loc["log_net_price"][0]),
                    "ci_high": float(model.conf_int().loc["log_net_price"][1]),
                    "n_obs": int(model.nobs),
                    "r_squared": float(model.rsquared),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[elasticity_model] skip {by}={level}: {exc}")
    return results


def estimate_by_segment_region(df: pd.DataFrame) -> list[dict]:
    """Elasticity by segment x region cells where enough data exists."""
    formula = (
        "log_units ~ log_net_price + discount_pct + marketing_campaign_flag "
        "+ log_competitor_index + log_category_index + log_inflation "
        "+ C(product_category) + C(product_id) + C(month)"
    )
    out: list[dict] = []
    for (seg, reg), sub in df.groupby(["customer_segment", "region"]):
        if len(sub) < 80 or sub["log_net_price"].nunique() < 12:
            continue
        try:
            model = smf.ols(formula, data=sub).fit(cov_type="HC3")
            out.append(
                {
                    "scope": "segment_region",
                    "group": f"{seg} | {reg}",
                    "customer_segment": seg,
                    "region": reg,
                    "elasticity": float(model.params["log_net_price"]),
                    "std_error": float(model.bse["log_net_price"]),
                    "p_value": float(model.pvalues["log_net_price"]),
                    "n_obs": int(model.nobs),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[elasticity_model] skip {seg}/{reg}: {exc}")
    return out


def estimate_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Elasticity within each K-Means cluster (df must contain `cluster_name`)."""
    if "cluster_name" not in df.columns:
        return pd.DataFrame()
    formula = (
        "log_units ~ log_net_price + discount_pct + marketing_campaign_flag "
        "+ log_competitor_index + log_category_index + log_inflation "
        "+ C(region) + C(product_category) + C(product_id) + C(month)"
    )
    rows: list[dict] = []
    for name, sub in df.groupby("cluster_name"):
        if len(sub) < 80 or sub["log_net_price"].nunique() < 12:
            continue
        try:
            m = smf.ols(formula, data=sub).fit(cov_type="HC3")
            rows.append({
                "cluster_name": str(name),
                "elasticity": float(m.params["log_net_price"]),
                "std_error": float(m.bse["log_net_price"]),
                "p_value": float(m.pvalues["log_net_price"]),
                "n_obs": int(m.nobs),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[elasticity_model] skip cluster {name}: {exc}")
    return pd.DataFrame(rows).sort_values("elasticity") if rows else pd.DataFrame()


def run_elasticity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (full_results_df, segment_region_df, overall_dict)."""
    overall = estimate_overall(df)
    rows = [overall]
    rows += _estimate_by(df, "customer_segment", "customer_segment")
    rows += _estimate_by(df, "region", "region")
    rows += _estimate_by(df, "product_category", "product_category")

    results_df = pd.DataFrame(rows)
    results_df["is_elastic"] = results_df["elasticity"] < config.ELASTIC_THRESHOLD
    results_df["is_significant"] = results_df["p_value"] < config.ELASTICITY_CONFIDENCE_PMAX

    seg_region_df = pd.DataFrame(estimate_by_segment_region(df))

    print(
        f"[elasticity_model] overall elasticity = {overall['elasticity']:.3f} "
        f"(R^2={overall['r_squared']:.3f}, n={overall['n_obs']:,})"
    )
    return results_df, seg_region_df, overall

# Dynamic Pricing & Elasticity Optimization

End-to-end pricing analytics portfolio project for **AquaPure Systems** (a synthetic,
anonymized water-treatment company). It estimates the **price elasticity of demand**
from historical sales, segments customers with **K-Means**, and produces
**margin-safe, business-ready pricing recommendations** through **constrained
optimization**.

> **Synthetic-data disclaimer.** This project uses synthetic data and anonymized
> company/product names. It is inspired by a real pricing analytics use case but
> does not disclose confidential company data, internal systems, customer
> information, or proprietary business results.

---

## Project Overview

Pricing decisions are hard because demand reacts to many forces at once — own price,
discounts, seasonality, competitors, customer mix, regional purchasing power, and
marketing. A naïve price-vs-volume correlation is misleading. This project uses an
**econometric log-log demand model with fixed effects** to isolate the causal
own-price elasticity, then turns those elasticities into concrete, constrained price
moves with quantified revenue and gross-profit impact.

## Business Problem

Finance and Marketing need visibility into how pricing changes affect demand across
**regions, product categories, and customer segments**, so they can answer:

- Which products and segments are price-sensitive (and which have pricing power)?
- Where does discounting create incremental demand, and where does it just leak margin?
- Which regions can support price increases?
- How should prices be optimized to improve revenue and gross margin?

## Dataset Description

A **synthetic monthly panel** spanning **3 years (36 periods)** across 5 regions,
5 customer segments, 5 product categories, and 30 products — **27,000 rows**.
Generated in `src/data_generation.py` with realistic embedded relationships
(downward-sloping demand, heterogeneous elasticity by segment/region/category,
marketing lift, competitor substitution, seasonality, macro effects, and random
noise). Key fields:

`date, year, month, period, region, customer_segment, product_category, product_id,
units_sold, list_price, net_price, discount_pct, cost_per_unit, gross_margin_pct,
revenue, gross_profit, marketing_campaign_flag, marketing_spend,
competitor_price_index, category_market_index, inflation_index, season_index,
order_frequency`.

The generator embeds **exogenous list-price shocks** so that price variation is not
purely a function of discounting — this is what makes elasticity statistically
identifiable.

## Methodology

**1. Log-log demand regression (the core model).**

```
log(units) = b0 + b1·log(net_price) + b2·discount_pct + b3·marketing_flag
           + b4·log(competitor_index) + b5·log(category_index) + b6·log(inflation)
           + region FE + product_category FE + product FE
           + customer_segment FE + month FE + e
```

`b1` is the **own-price elasticity of demand**. The model uses **HC3 robust standard
errors**. Fixed effects for region, category, product, segment, and month remove
confounding from seasonality, regional purchasing power, product mix, and campaign
timing — so the result is not a naïve price-volume correlation. Elasticity is
re-estimated within each segment, region, and product category, and within
segment×region cells where data supports it.

**2. K-Means customer segmentation** over behavioural features (average order value,
units, discount received, order frequency, margin contribution, net price), with
business-friendly cluster names.

**3. Competitor / cross-price effects** via category-level competitor price indices,
keeping the model commercially useful without multicollinearity from modelling every
substitute directly.

**4. Constrained price optimization** per category×segment×region cell. Demand
response is `new_demand = current_demand · (new_price / current_price) ^ elasticity`.
Price changes are capped at **±10%**, with a **30% gross-margin floor**, optimizing
gross profit (configurable to revenue). Low-confidence elasticities are flagged for
manual review.

## Project Structure

```
dynamic-pricing-elasticity/
├── data/
│   └── synthetic_pricing_data.csv
├── outputs/
│   ├── elasticity_results.csv
│   ├── segment_elasticity_summary.csv
│   ├── pricing_recommendations.csv
│   ├── optimization_results.csv
│   ├── business_impact_recommendations.md
│   └── charts/
│       ├── elasticity_by_segment.png
│       ├── elasticity_by_region.png
│       ├── revenue_margin_simulation.png
│       ├── recommended_price_changes.png
│       └── segmentation_clusters.png
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── elasticity_model.py
│   ├── segmentation.py
│   ├── optimization.py
│   ├── recommendations.py
│   ├── visualization.py
│   └── presentation.py
├── notebooks/
│   └── dynamic_pricing_analysis.ipynb
├── presentation/
│   └── pitch_deck_dynamic_pricing.pdf
├── main.py
├── requirements.txt
└── README.md
```

## How to Run the Project

```bash
pip install -r requirements.txt
python main.py
```

This regenerates the dataset, fits the models, runs optimization, and writes all
CSVs, charts, the business recommendations markdown, and the PDF pitch deck. Outputs
are reproducible via the fixed random seed in `src/config.py`.

## Key Outputs

- `outputs/elasticity_results.csv` — elasticity overall and by segment/region/category
  with robust SEs, p-values, and confidence intervals.
- `outputs/segment_elasticity_summary.csv` — segment elasticity joined to K-Means clusters.
- `outputs/optimization_results.csv` — recommended price per cell with revenue / GP impact.
- `outputs/pricing_recommendations.csv` — prioritised, client-facing recommendation list.
- `outputs/business_impact_recommendations.md` — narrative tied to the actual numbers.
- `outputs/charts/*.png` — elasticity, segmentation, simulation, and recommendation charts.
- `presentation/pitch_deck_dynamic_pricing.pdf` — 18-slide client-facing executive deck (every figure backed by a chart and numeric takeaway).
- `presentation/AquaPure_Pricing_Technical_Briefing.docx` — detailed presenter's companion: full methodology, the complete modelling-iteration log, results, assumptions, and a client Q&A playbook.

## Example Business Recommendations

From the latest run (synthetic data):

- **Overall own-price elasticity ≈ −1.05** (robust SE ≈ 0.01, R² ≈ 0.88).
- **Strongest pricing power:** Industrial Accounts (≈ −0.78) and the North region (≈ −0.78);
  **most price-sensitive:** Homeowners (≈ −1.41) and the West region (≈ −1.41).
- **Most elastic category:** Residential Filters (≈ −1.65); **least elastic:** Industrial
  Treatment (≈ −0.70).
- Constrained gross-profit optimization yields **≈ +10.6% gross profit** for a small
  (≈ −1.4%) revenue trade-off — the classic signature of exchanging a little volume for
  materially higher margin, all within ±10% price moves and a 30% margin floor.

See `outputs/business_impact_recommendations.md` for the full, data-driven writeup.

## Skills Demonstrated

Python · SQL-style analytical thinking · data preprocessing · econometric modeling ·
price elasticity estimation · K-Means clustering · constrained optimization ·
business storytelling · pricing strategy · executive reporting.

---

_Built as a portfolio project. All data is synthetic; see disclaimer above._

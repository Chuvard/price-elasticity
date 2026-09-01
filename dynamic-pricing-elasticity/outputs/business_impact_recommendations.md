# Business Impact & Pricing Recommendations — AquaPure Systems

_Generated from model outputs. This project uses synthetic data and anonymized company/product names. It is inspired by a real pricing analytics use case but does not disclose confidential company data, internal systems, customer information, or proprietary business results._

## Executive Summary

Across the analysed portfolio, the estimated **overall own-price elasticity of demand is -1.01** (robust SE 0.01, R² 0.80, n = 26,987). A value of -1.01 means a 1% increase in net price is associated with roughly a 1.01% change in demand, holding seasonality, marketing, competitor pricing, region, product mix and customer segment constant via fixed effects.

Constrained optimization (objective: **gross profit**, price moves capped at ±10%, margin floor 30%) identifies **$231,795 of incremental gross profit (+10.1%)** over the analysed horizon, against a baseline of $2,297,370 gross profit. This comes with a modest revenue trade-off of **-0.9%** ($-47,509 on a $5,156,543 base) — the expected signature of trading a little volume for materially higher margin, all within governance constraints.

## Where the Pricing Power Is

- **Lowest elasticity (strongest pricing power) segment:** `Industrial Accounts` at **-0.46** — these buyers tolerate price increases with little volume loss.
- **Highest elasticity (most price-sensitive) segment:** `Homeowners` at **-1.45** — protect volume here; use targeted promotions rather than blanket increases.
- **Region with strongest pricing power:** `North` (-0.73); most sensitive region: `West` (-1.44).
- **Category with most pricing power:** `Industrial Treatment` (-0.62); most elastic category: `Residential Filters` (-1.61).

## Recommended Actions

The optimizer produced **55 price-increase**, **0 discount-reduction**, **1 targeted-promotion**, and **0 manual-review** recommendations across 125 product-category × segment × region cells.

### Top 5 gross-profit opportunities

| Category | Segment | Region | Action | Price Δ% | GP impact |
|---|---|---|---|---:|---:|
| Industrial Treatment | Distributors | North | Increase price | +10.0% | $6,538 |
| Industrial Treatment | Distributors | South | Hold price | +10.0% | $5,547 |
| Industrial Treatment | Small Business | North | Increase price | +10.0% | $5,383 |
| Industrial Treatment | Distributors | Northeast | Hold price | +10.0% | $5,332 |
| Industrial Treatment | Homeowners | North | Hold price | +10.0% | $5,204 |

### Where discounts should be reduced (margin leakage)

- No material discount-reduction cells flagged at current thresholds.

### Where targeted promotions should be used (elastic cells)

- **Service Contracts / Homeowners / West**: elasticity -1.91; a targeted price reduction of -10.0% is expected to lift demand +22.2%.

## Practical Next Steps

**Finance:** lock margin floors per category, fund the price-increase cells above first (lowest execution risk, highest GP), and require model sign-off for any change beyond ±10%.

**Marketing:** redirect promotional budget away from low-elasticity segments (e.g. `Industrial Accounts`) toward the elastic cells listed above, where discounts actually buy incremental volume.

**Sales:** use the recommended price bands as guardrails in quoting tools; escalate manual-review cells rather than discounting on instinct.

## How a Client Can Use This Approach

- **Replace blanket discounting with targeted discounting** driven by measured segment elasticity instead of habit.
- **Protect margin in low-elasticity segments** by holding or raising price where volume is insensitive.
- **Identify product categories with pricing power** and prioritise them for list-price review.
- **Simulate price changes before implementation** using the demand-response equation, avoiding costly live experiments.
- **Align Finance, Marketing, and Sales** around one shared, data-driven set of pricing rules.
- **Build a recurring pricing governance process** — monthly re-estimation, constraint review, and a single approval workflow.

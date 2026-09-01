"""Central configuration for the Dynamic Pricing & Elasticity Optimization project.

All tunable parameters, file paths, and business assumptions live here so the
rest of the pipeline stays clean and reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Privacy / branding (synthetic only — no real company data)
# --------------------------------------------------------------------------- #
SYNTHETIC_COMPANY_NAME: str = "AquaPure Systems"
DISCLAIMER: str = (
    "This project uses synthetic data and anonymized company/product names. "
    "It is inspired by a real pricing analytics use case but does not disclose "
    "confidential company data, internal systems, customer information, or "
    "proprietary business results."
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
CHART_DIR: Path = OUTPUT_DIR / "charts"
PRESENTATION_DIR: Path = ROOT_DIR / "presentation"

DATA_PATH: Path = DATA_DIR / "synthetic_pricing_data.csv"
ELASTICITY_RESULTS_PATH: Path = OUTPUT_DIR / "elasticity_results.csv"
SEGMENT_ELASTICITY_PATH: Path = OUTPUT_DIR / "segment_elasticity_summary.csv"
PRICING_RECS_PATH: Path = OUTPUT_DIR / "pricing_recommendations.csv"
OPTIMIZATION_PATH: Path = OUTPUT_DIR / "optimization_results.csv"
BUSINESS_MD_PATH: Path = OUTPUT_DIR / "business_impact_recommendations.md"
PITCH_DECK_PATH: Path = PRESENTATION_DIR / "pitch_deck_dynamic_pricing.pdf"


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (DATA_DIR, OUTPUT_DIR, CHART_DIR, PRESENTATION_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Synthetic data design
# --------------------------------------------------------------------------- #
START_YEAR: int = 2023
N_PERIODS: int = 36  # 36 monthly periods => 3 years of data

REGIONS: list[str] = ["North", "South", "Midwest", "West", "Northeast"]

# True (latent) regional pricing power: a multiplier applied to baseline
# elasticity. Values < 1 mean LESS elastic (stronger pricing power).
REGION_PRICING_POWER: dict[str, float] = {
    "North": 0.78,      # strongest pricing power (least elastic)
    "Northeast": 0.88,
    "Midwest": 1.00,
    "South": 1.18,
    "West": 1.32,       # most price-sensitive
}

PRODUCT_CATEGORIES: list[str] = [
    "Residential Filters",
    "Commercial Systems",
    "Industrial Treatment",
    "Replacement Parts",
    "Service Contracts",
]

# Baseline (latent) own-price elasticity by product category (negative).
CATEGORY_BASE_ELASTICITY: dict[str, float] = {
    "Residential Filters": -1.65,   # very elastic, commoditized
    "Commercial Systems": -1.10,
    "Industrial Treatment": -0.72,  # least elastic, specialized
    "Replacement Parts": -0.85,
    "Service Contracts": -0.95,
}

# Typical unit cost as a fraction of list price, by category.
CATEGORY_COST_RATIO: dict[str, float] = {
    "Residential Filters": 0.62,
    "Commercial Systems": 0.58,
    "Industrial Treatment": 0.55,
    "Replacement Parts": 0.45,
    "Service Contracts": 0.40,
}

# Approximate list-price level (USD) per category. Sized for a small regional
# water-treatment dealer (consumer filters, parts and service plans) so that
# aggregate dollar figures stay realistic rather than ballooning into billions.
CATEGORY_LIST_PRICE: dict[str, float] = {
    "Residential Filters": 33.0,
    "Commercial Systems": 116.0,
    "Industrial Treatment": 211.0,
    "Replacement Parts": 11.6,
    "Service Contracts": 129.0,
}

# Baseline monthly units per (region x segment) per product. Small integers keep
# the company realistically sized; zero-demand rows are dropped (sparse panel).
CATEGORY_BASE_UNITS: dict[str, float] = {
    "Residential Filters": 2.0,
    "Commercial Systems": 2.0,
    "Industrial Treatment": 1.8,
    "Replacement Parts": 3.0,
    "Service Contracts": 1.6,
}

# Global demand multiplier — single lever to scale overall business size.
DEMAND_SCALE: float = 0.90

CUSTOMER_SEGMENTS: list[str] = [
    "Homeowners",
    "Small Business",
    "Municipal",
    "Industrial Accounts",
    "Distributors",
]

# Segment-level elasticity multiplier (lower = more pricing power).
SEGMENT_PRICING_POWER: dict[str, float] = {
    "Homeowners": 1.30,          # most price sensitive
    "Distributors": 1.15,
    "Small Business": 1.00,
    "Municipal": 0.80,
    "Industrial Accounts": 0.70, # least price sensitive
}

N_PRODUCTS_PER_CATEGORY: int = 6

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
N_CLUSTERS: int = 4
CLUSTER_FEATURES: list[str] = [
    "avg_order_value",
    "avg_units_sold",
    "avg_discount_received",
    "order_frequency",
    "gross_margin_contribution",
    "avg_net_price",
]
BUSINESS_CLUSTER_NAMES: list[str] = [
    "Price-Sensitive Buyers",
    "Premium / Service-Oriented Buyers",
    "Promotion-Responsive Buyers",
    "Stable High-Margin Customers",
    "Low-Volume Opportunistic Buyers",
]

# --------------------------------------------------------------------------- #
# Optimization constraints
# --------------------------------------------------------------------------- #
MAX_PRICE_CHANGE: float = 0.10          # +/- 10%
MIN_MARGIN_FLOOR: float = 0.30          # never let gross margin drop below 30%
OPTIMIZATION_OBJECTIVE: str = "gross_profit"  # "revenue" or "gross_profit"
ELASTICITY_CONFIDENCE_PMAX: float = 0.10      # p-value threshold for "stable"
ELASTIC_THRESHOLD: float = -1.0               # |E| > 1 => elastic

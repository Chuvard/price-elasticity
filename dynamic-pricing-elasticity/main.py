"""End-to-end entry point for the Dynamic Pricing & Elasticity project.

Run with:  python main.py

Pipeline:
  1. Generate synthetic pricing dataset
  2. Clean / preprocess / engineer features
  3. Estimate price elasticity (overall + by segment/region/category)
  4. K-Means customer segmentation
  5. Constrained price optimization
  6. Build pricing recommendations + business markdown
  7. Generate charts
  8. Build PDF pitch deck
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python main.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (
    config,
    data_generation,
    elasticity_model,
    optimization,
    preprocessing,
    presentation,
    recommendations,
    segmentation,
    visualization,
)


def main() -> None:
    print("=" * 70)
    print("Dynamic Pricing & Elasticity Optimization —", config.SYNTHETIC_COMPANY_NAME)
    print("=" * 70)
    config.ensure_dirs()

    # 1. Synthetic data
    data_generation.generate_and_save()

    # 2. Preprocess
    df = preprocessing.preprocess()

    # 3. Elasticity
    results_df, seg_region_df, overall = elasticity_model.run_elasticity(df)
    results_df.to_csv(config.ELASTICITY_RESULTS_PATH, index=False)
    print(f"[main] wrote {config.ELASTICITY_RESULTS_PATH}")

    # 4. Segmentation
    profiles = segmentation.run_segmentation(df)
    df_clustered = segmentation.attach_clusters(df, profiles)
    cluster_elast = elasticity_model.estimate_by_cluster(df_clustered)
    if not cluster_elast.empty:
        cluster_elast.to_csv(config.OUTPUT_DIR / "cluster_elasticity.csv", index=False)
        print(f"[main] wrote {config.OUTPUT_DIR / 'cluster_elasticity.csv'}")

    # 5. Optimization
    opt_df = optimization.run_optimization(
        df, results_df, seg_region_df, overall["elasticity"]
    )
    opt_df.to_csv(config.OPTIMIZATION_PATH, index=False)
    print(f"[main] wrote {config.OPTIMIZATION_PATH}")

    # 6. Recommendations + business markdown + segment summary
    recommendations.build_pricing_recommendations(opt_df)
    recommendations.write_segment_elasticity_summary(results_df, profiles)
    recommendations.write_business_markdown(results_df, seg_region_df, opt_df, overall)

    # 7. Charts
    charts = visualization.generate_all_charts(
        df, results_df, opt_df, profiles, overall["elasticity"], cluster_elast
    )

    # 8. PDF pitch deck
    presentation.build_pitch_deck(
        overall, results_df, opt_df, charts, profiles, cluster_elast
    )

    print("=" * 70)
    print("DONE. Key outputs in outputs/ and presentation/.")
    print(
        f"Overall elasticity: {overall['elasticity']:.3f} | "
        f"Total GP impact: ${opt_df['expected_gross_profit_impact'].sum():,.0f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

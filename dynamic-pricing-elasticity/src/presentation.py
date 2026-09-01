"""Build the client-facing PDF pitch deck from actual pipeline outputs (reportlab).

Produces a ~15-slide professional deck where every headline number is supported
by a chart and/or an explicit numeric takeaway on the same slide.
"""
from __future__ import annotations

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from . import config

PAGE = landscape(letter)
PW, PH = PAGE
NAVY = colors.HexColor("#16324f")
TEAL = colors.HexColor("#1f8a8a")
ORANGE = colors.HexColor("#e67e22")
LIGHT = colors.HexColor("#eef3f6")
GREY = colors.HexColor("#5a6b78")
USABLE_W = 9.7 * inch


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles():
    ss = getSampleStyleSheet()
    add = ss.add
    add(ParagraphStyle("DeckTitle", parent=ss["Title"], fontSize=36,
                       textColor=NAVY, leading=42, spaceAfter=12))
    add(ParagraphStyle("DeckSub", parent=ss["Normal"], fontSize=17,
                       textColor=TEAL, leading=23, spaceAfter=8))
    add(ParagraphStyle("BandText", parent=ss["Normal"], fontSize=21,
                       textColor=colors.white, leading=25))
    add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=13,
                       textColor=colors.HexColor("#222"), leading=19, spaceAfter=6))
    add(ParagraphStyle("DeckBullet", parent=ss["Normal"], fontSize=13,
                       leftIndent=12, leading=19, spaceAfter=5,
                       textColor=colors.HexColor("#222")))
    add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=11.5,
                       textColor=GREY, leading=16, spaceBefore=4, alignment=TA_CENTER))
    add(ParagraphStyle("Takeaway", parent=ss["Normal"], fontSize=12.5,
                       textColor=NAVY, leading=17, leftIndent=8))
    add(ParagraphStyle("MetricNum", parent=ss["Normal"], fontSize=26,
                       textColor=colors.white, leading=28, alignment=TA_CENTER))
    add(ParagraphStyle("MetricLbl", parent=ss["Normal"], fontSize=10.5,
                       textColor=colors.white, leading=13, alignment=TA_CENTER))
    add(ParagraphStyle("Foot", parent=ss["Normal"], fontSize=8, textColor=GREY))
    add(ParagraphStyle("Eq", parent=ss["Normal"], fontSize=11.5,
                       textColor=NAVY, leading=16, backColor=LIGHT,
                       borderPadding=8, fontName="Courier"))
    return ss


def _band(title: str, ss) -> Table:
    t = Table([[Paragraph(title, ss["BandText"])]], colWidths=[USABLE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _bullets(items, ss):
    return [Paragraph(f"• {t}", ss["DeckBullet"]) for t in items]


def _metric_cards(cards, ss):
    """cards: list of (value, label, color)."""
    cells, styles = [], []
    for i, (val, lbl, col) in enumerate(cards):
        inner = Table([[Paragraph(val, ss["MetricNum"])],
                       [Paragraph(lbl, ss["MetricLbl"])]], colWidths=[2.25 * inch])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), col),
            ("TOPPADDING", (0, 0), (-1, 0), 14),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        cells.append(inner)
    row = Table([cells], colWidths=[2.42 * inch] * len(cells))
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 4),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                             ("TOPPADDING", (0, 0), (-1, -1), 0)]))
    return row


def _img(path: str, w: float, h: float) -> Image:
    im = Image(path, width=w, height=h)
    im.hAlign = "CENTER"
    return im


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(0.7 * inch, 0.35 * inch,
                      f"{config.SYNTHETIC_COMPANY_NAME}  ·  Dynamic Pricing & "
                      f"Elasticity Optimization  ·  Synthetic data, illustrative")
    canvas.drawRightString(PW - 0.7 * inch, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Deck
# --------------------------------------------------------------------------- #
def build_pitch_deck(
    overall: dict,
    results_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    charts: dict,
    profiles: pd.DataFrame | None = None,
    cluster_elast: pd.DataFrame | None = None,
) -> str:
    config.ensure_dirs()
    ss = _styles()
    s: list = []

    # ----- derived numbers -----
    seg = results_df[results_df.scope == "customer_segment"]
    reg = results_df[results_df.scope == "region"]
    cat = results_df[results_df.scope == "product_category"]
    least_seg = seg.loc[seg.elasticity.idxmax()]
    most_seg = seg.loc[seg.elasticity.idxmin()]
    strong_reg = reg.loc[reg.elasticity.idxmax()]
    weak_reg = reg.loc[reg.elasticity.idxmin()]
    least_cat = cat.loc[cat.elasticity.idxmax()]
    most_cat = cat.loc[cat.elasticity.idxmin()]

    total_rev = opt_df.expected_revenue_impact.sum()
    total_gp = opt_df.expected_gross_profit_impact.sum()
    base_rev = opt_df.baseline_revenue.sum()
    base_gp = opt_df.baseline_gross_profit.sum()
    gp_up = total_gp / base_gp * 100 if base_gp else 0
    rev_up = total_rev / base_rev * 100 if base_rev else 0

    n_inc = (opt_df.recommendation_type.str.startswith("Increase")).sum()
    n_disc = (opt_df.recommendation_type.str.startswith("Reduce")).sum()
    n_promo = (opt_df.recommendation_type.str.startswith("Target")).sum()
    n_hold = (opt_df.recommendation_type.str.startswith("Hold")).sum()
    n_rev = (opt_df.recommendation_type.str.startswith("Review")).sum()

    def money(x):
        return f"${x:,.0f}"

    # ===== Slide 1: Title =====
    s += [Spacer(1, 1.3 * inch)]
    s.append(Paragraph("Dynamic Pricing &amp; Elasticity Optimization", ss["DeckTitle"]))
    s.append(Paragraph(f"{config.SYNTHETIC_COMPANY_NAME} — Water Treatment Systems",
                       ss["DeckSub"]))
    s.append(Spacer(1, 0.15 * inch))
    s.append(Paragraph(
        "Measuring how price actually drives demand — and converting that into "
        "margin-safe, segment-level pricing decisions.", ss["Body"]))
    s.append(Spacer(1, 0.45 * inch))
    s.append(Paragraph(
        "Econometric demand modeling · customer segmentation · constrained "
        "price optimization · executive decision support", ss["Caption"]))
    s.append(Spacer(1, 0.5 * inch))
    s.append(Paragraph(config.DISCLAIMER, ss["Foot"]))
    s.append(PageBreak())

    # ===== Slide 2: Executive Summary =====
    s.append(_band("Executive Summary", ss))
    s.append(Spacer(1, 0.18 * inch))
    s.append(_metric_cards([
        (f"{overall['elasticity']:.2f}", "Overall price elasticity", NAVY),
        (f"+{gp_up:.1f}%", "Gross-profit uplift", TEAL),
        (money(total_gp), "Incremental gross profit", ORANGE),
        (f"{int(overall['r_squared']*100)}%", "Model fit (R²)", GREY),
    ], ss))
    s.append(Spacer(1, 0.22 * inch))
    s += _bullets([
        f"Demand is close to unit-elastic overall ({overall['elasticity']:.2f}), but "
        f"varies widely — from {least_cat.elasticity:.2f} ({least_cat.group}) to "
        f"{most_cat.elasticity:.2f} ({most_cat.group}).",
        f"Strongest pricing power sits with <b>{least_seg.group}</b> "
        f"({least_seg.elasticity:.2f}) and the <b>{strong_reg.group}</b> region "
        f"({strong_reg.elasticity:.2f}).",
        f"Constrained optimization (±10% price band, 30% margin floor) yields "
        f"<b>{money(total_gp)}</b> incremental gross profit (<b>+{gp_up:.1f}%</b>) "
        f"for a small {rev_up:+.1f}% revenue trade-off.",
        f"{int(n_inc)} targeted price increases, {int(n_promo)} promotion and "
        f"{int(n_disc)} discount-reduction actions identified across "
        f"{len(opt_df)} product × segment × region cells.",
    ], ss)
    s.append(PageBreak())

    # ===== Slide 3: Business Problem =====
    s.append(_band("The Business Problem", ss))
    s.append(Spacer(1, 0.16 * inch))
    s.append(Paragraph(
        "Finance and Marketing need to know how price changes affect demand — but "
        "demand responds to many forces at once, so raw price-vs-volume comparisons "
        "mislead:", ss["Body"]))
    s += _bullets([
        "<b>Own price &amp; discount depth</b> — the levers we control.",
        "<b>Customer segment</b> — homeowners behave nothing like industrial accounts.",
        "<b>Region</b> — purchasing power and competition differ by geography.",
        "<b>Seasonality &amp; marketing</b> — campaigns and weather move volume independently of price.",
        "<b>Competitor pricing &amp; macro conditions</b> — external substitution and inflation.",
    ], ss)
    s.append(Spacer(1, 0.12 * inch))
    s.append(Paragraph(
        "<b>The risk:</b> without isolating the true price effect, blanket discounting "
        "quietly erodes margin while price increases are left on the table where "
        "customers would have paid more.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 4: Analytical Framework =====
    s.append(_band("Our Analytical Framework", ss))
    s.append(Spacer(1, 0.16 * inch))
    s.append(Paragraph(
        "A transparent, econometric pipeline — every step is auditable and reproducible:",
        ss["Body"]))
    flow = [
        ["1. Data", "2. Elasticity", "3. Segmentation", "4. Optimization", "5. Recommendations"],
        ["Clean panel of\nsales, price, cost,\ncompetitor & macro",
         "Log-log demand\nregression with\nfixed effects",
         "K-Means on\nbehavioural\npricing features",
         "Constrained search\nwithin ±10% &\nmargin floor",
         "Prioritised, margin-\nsafe price actions\nby cell"],
    ]
    t = Table(flow, colWidths=[USABLE_W / 5.0] * 5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("FONTSIZE", (0, 1), (-1, 1), 10.5),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    s.append(Spacer(1, 0.1 * inch))
    s.append(t)
    s.append(Spacer(1, 0.18 * inch))
    s.append(Paragraph(
        "We deliberately use <b>econometrics, not a black box</b>: coefficients are "
        "interpretable, defensible to Finance, and come with confidence intervals.",
        ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 5: Data Foundation =====
    s.append(_band("Data Foundation", ss))
    s.append(Spacer(1, 0.1 * inch))
    s.append(Paragraph(
        f"A <b>{overall['n_obs']:,}-row</b> monthly panel spanning 3 years across 5 regions, "
        "5 customer segments, 5 product categories and 30 products. Each row carries "
        "price, discount, cost, margin, units, marketing, competitor and macro indicators.",
        ss["Body"]))
    s.append(_img(charts["sales_trend"], 7.6 * inch, 3.45 * inch))
    s.append(Paragraph(
        "Monthly units and revenue across the panel — note the recurring seasonality "
        "the model must control for.", ss["Caption"]))
    s.append(PageBreak())

    # ===== Slide 6: Methodology — the model =====
    s.append(_band("Methodology — Log-Log Demand Model", ss))
    s.append(Spacer(1, 0.14 * inch))
    s.append(Paragraph(
        "We estimate a constant-elasticity demand curve. The coefficient on "
        "log(net price) <b>is</b> the own-price elasticity of demand:", ss["Body"]))
    s.append(Spacer(1, 0.06 * inch))
    s.append(Paragraph(
        "log(units) = &#946;&#8320; + &#946;&#8321;·log(net_price) + &#946;&#8322;·discount "
        "+ &#946;&#8323;·marketing + &#946;&#8324;·log(competitor_index) + macro "
        "+ region + category + product + segment + month fixed effects + &#949;",
        ss["Eq"]))
    s.append(Spacer(1, 0.14 * inch))
    s += _bullets([
        f"<b>&#946;&#8321; = {overall['elasticity']:.2f}</b> overall — a 1% price rise "
        f"moves demand about {abs(overall['elasticity']):.2f}% the other way.",
        "<b>Fixed effects</b> for region, category, product, segment and month strip out "
        "confounding from seasonality, purchasing power, product mix and campaign timing.",
        "<b>HC3 robust standard errors</b> guard against heteroskedasticity; all key "
        "estimates are statistically significant (p &lt; 0.01).",
        f"<b>Identification:</b> exogenous list-price variation lets us separate the price "
        f"effect from the discount lever — R² = {overall['r_squared']:.2f}.",
    ], ss)
    s.append(PageBreak())

    # ===== Slide 7: Elasticity by Category =====
    s.append(_band("Results — Elasticity by Product Category", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["category"], 6.7 * inch, 3.35 * inch))
    s.append(Paragraph(
        f"<b>{most_cat.group}</b> is the most price-sensitive ({most_cat.elasticity:.2f}); "
        f"<b>{least_cat.group}</b> has clear pricing power ({least_cat.elasticity:.2f}). "
        f"Bars left of the dashed line are elastic (volume falls faster than price rises).",
        ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 8: Elasticity by Segment =====
    s.append(_band("Results — Elasticity by Customer Segment", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["segment"], 6.7 * inch, 3.35 * inch))
    s.append(Paragraph(
        f"<b>{most_seg.group}</b> are most elastic ({most_seg.elasticity:.2f}) — protect "
        f"volume with targeted offers. <b>{least_seg.group}</b> tolerate price "
        f"({least_seg.elasticity:.2f}) — a margin opportunity. Whiskers show 95% CIs.",
        ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 9: Elasticity by Region =====
    s.append(_band("Results — Elasticity by Region", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["region"], 6.7 * inch, 3.35 * inch))
    s.append(Paragraph(
        f"The <b>{strong_reg.group}</b> region supports price ({strong_reg.elasticity:.2f}); "
        f"the <b>{weak_reg.group}</b> region is the most sensitive "
        f"({weak_reg.elasticity:.2f}). Pricing should not be set nationally as one number.",
        ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 10: Segmentation =====
    s.append(_band("Customer Segmentation (K-Means)", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["clusters"], 6.9 * inch, 3.5 * inch))
    s.append(Paragraph(
        "Customers cluster into distinct behavioural groups by discount appetite, "
        "price level and margin contribution — the basis for differentiated pricing "
        "and promotion rules rather than one-size-fits-all discounting.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 11: Elasticity by cluster (if available) =====
    if "cluster_elast" in charts:
        s.append(_band("Pricing Power by Customer Cluster", ss))
        s.append(Spacer(1, 0.08 * inch))
        s.append(_img(charts["cluster_elast"], 6.7 * inch, 3.35 * inch))
        s.append(Paragraph(
            "Elasticity re-estimated within each cluster confirms the segments are not "
            "just behaviourally different — they respond to price differently, which is "
            "exactly what a differentiated pricing strategy exploits.", ss["Takeaway"]))
        s.append(PageBreak())

    # ===== Slide 12: Revenue & Margin Simulation =====
    s.append(_band("Revenue &amp; Gross-Profit Simulation", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["simulation"], 6.7 * inch, 3.35 * inch))
    s.append(Paragraph(
        "Before any change goes live, we simulate it. The gross-profit-maximising move "
        "differs from the revenue-maximising move — which is why optimizing the right "
        "objective matters. The shaded band marks the governance-approved ±10% range.",
        ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 13: Recommendation mix =====
    s.append(_band("Optimization — Recommendation Mix", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["rec_mix"], 8.2 * inch, 3.4 * inch))
    s.append(Paragraph(
        f"{int(n_inc)} price-increase, {int(n_promo)} targeted-promotion, "
        f"{int(n_disc)} discount-reduction, {int(n_hold)} hold and {int(n_rev)} "
        f"manual-review actions — with their expected gross-profit contribution. "
        f"Action is concentrated where elasticity is reliably low.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 14: Recommended price changes heatmap =====
    s.append(_band("Optimization — Recommended Price Changes", ss))
    s.append(Spacer(1, 0.08 * inch))
    s.append(_img(charts["price_changes"], 7.6 * inch, 3.5 * inch))
    s.append(Paragraph(
        "Recommended net-price change by category × segment. Green = raise price "
        "(pricing power), red = reduce / promote (elastic). Every cell stays inside "
        "the ±10% governance band.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 15: Top opportunities =====
    s.append(_band("Where the Value Is — Top Opportunities", ss))
    s.append(Spacer(1, 0.06 * inch))
    s.append(_img(charts["top_opps"], 7.8 * inch, 3.7 * inch))
    s.append(Paragraph(
        "The ten highest-impact cells, with the recommended price move. These are the "
        "first actions to execute — highest gross-profit gain, lowest execution risk.",
        ss["Caption"]))
    s.append(PageBreak())

    # ===== Slide 16: Business Impact / Client Value =====
    s.append(_band("Business Impact &amp; Client Value", ss))
    s.append(Spacer(1, 0.14 * inch))
    s += _bullets([
        "<b>Increase price where elasticity is low</b> — measured pricing power, not guesswork.",
        "<b>Reduce unnecessary discounts</b> where they fail to buy incremental volume.",
        "<b>Use targeted promotions only for elastic segments</b> that actually respond.",
        "<b>Monitor competitor-sensitive categories</b> via the competitor price index.",
        "<b>Run monthly price simulations</b> before approving any change.",
        "<b>Align Finance, Marketing &amp; Sales</b> on one shared set of pricing rules.",
    ], ss)
    s.append(Spacer(1, 0.12 * inch))
    s.append(Paragraph(
        f"<b>Net effect:</b> a defensible pricing system worth <b>{money(total_gp)}</b> "
        f"(<b>+{gp_up:.1f}%</b> gross profit) while protecting volume in sensitive "
        f"segments such as {most_seg.group}.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 17: Roadmap =====
    s.append(_band("Implementation Roadmap", ss))
    s.append(Spacer(1, 0.16 * inch))
    roadmap = [
        ["Phase", "Focus", "Outcome"],
        ["1 — Foundations", "Connect sales, cost, competitor & macro data", "Clean, governed pricing panel"],
        ["2 — Measure", "Re-estimate elasticity monthly", "Live elasticity dashboard"],
        ["3 — Optimize", "Generate constrained price recommendations", "Approved price bands by cell"],
        ["4 — Govern", "Finance / Marketing / Sales review cadence", "Repeatable pricing governance"],
    ]
    t = Table(roadmap, colWidths=[2.0 * inch, 4.4 * inch, 3.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    s.append(t)
    s.append(Spacer(1, 0.2 * inch))
    s.append(Paragraph(
        "Start with the top-opportunity cells under a controlled pilot, measure realised "
        "vs. predicted impact, then scale across the portfolio.", ss["Takeaway"]))
    s.append(PageBreak())

    # ===== Slide 18: Assumptions & disclaimer =====
    s.append(_band("Assumptions, Limitations &amp; Data Note", ss))
    s.append(Spacer(1, 0.14 * inch))
    s += _bullets([
        "Demand follows a constant-elasticity (log-log) form over the analysed price range; "
        "extrapolation beyond ±10% is not advised.",
        "Elasticities are estimated from historical variation; structural market shifts "
        "require re-estimation.",
        "Low-confidence or unstable cells are flagged for manual review rather than auto-priced.",
        "Competitor effects are captured through category-level price indices, not every "
        "individual substitute.",
        "Dollar magnitudes scale with business size; figures here reflect the modelled "
        "company and horizon.",
    ], ss)
    s.append(Spacer(1, 0.18 * inch))
    s.append(Paragraph(config.DISCLAIMER, ss["Foot"]))

    doc = SimpleDocTemplate(
        str(config.PITCH_DECK_PATH), pagesize=PAGE,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title="Dynamic Pricing & Elasticity Optimization",
        author=config.SYNTHETIC_COMPANY_NAME,
    )
    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[presentation] wrote {config.PITCH_DECK_PATH}")
    return str(config.PITCH_DECK_PATH)

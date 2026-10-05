"""Week 2: audit Week 1 Residential data, create EDA datasets and reports.

Run: python src/week2.py (requires pandas, numpy, matplotlib, plotly).
Use --no-plots to save data and summaries without opening graph windows.
Inputs are Week 1 CSVs in data/processed/week1; outputs go to data/processed/week2.
No deduplication, imputation, or outlier removal is performed.
"""

import re
import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from week1 import DATA_DIR, START_MONTH, END_MONTH, OUTPUT_DIR as INPUT_DIR

OUTPUT_DIR = INPUT_DIR.parent / "week2"
num_dist_field = [
    "ListingId", "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]

field_labels = {
    "ClosePrice": ("Sale price", "US dollars ($)"),
    "ListPrice": ("Asking price", "US dollars ($)"),
    "OriginalListPrice": ("Original asking price", "US dollars ($)"),
    "LivingArea": ("Living area", "square feet, per primer"),
    "LotSizeAcres": ("Lot size", "acres"),
    "BedroomsTotal": ("Bedrooms", "count"),
    "BathroomsTotalInteger": ("Bathrooms", "count"),
    "DaysOnMarket": ("Days on market", "days"),
    "YearBuilt": ("Year built", "calendar year"),
}

price_fields = {"ClosePrice", "ListPrice", "OriginalListPrice"}
PLOT_CONFIG = {"scrollZoom": False, "displaylogo": False}


def missing_val_check(df):
    """Audit input nulls before column selection or row filtering."""
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    })
    report["over_90_pct"] = report["missing_pct"] > 90
    return report.sort_values("missing_pct", ascending=False)


def raw_property_types():
    """Read only PropertyType from available raw files, without redoing Week 1."""
    rows = []
    for prefix, label in [("CRMLSListing", "Listings"), ("CRMLSSold", "Sold")]:
        pattern = re.compile(rf"{prefix}(\d{{6}})(?:_filled)?\.csv")
        for file in sorted(DATA_DIR.glob(f"{prefix}*.csv")):
            match = pattern.fullmatch(file.name)
            if not match or not START_MONTH <= match.group(1) <= END_MONTH:
                continue
            counts = pd.read_csv(file, usecols=["PropertyType"])["PropertyType"].value_counts(dropna=False)
            for value, count in counts.items():
                rows.append({"dataset": label, "PropertyType": value, "count": count})
    if not rows:
        return pd.DataFrame(columns=["dataset", "PropertyType", "count"])
    return pd.DataFrame(rows).groupby(
        ["dataset", "PropertyType"], dropna=False, as_index=False
    )["count"].sum()


def finite_values(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def distribution_figure(res_list_clean, res_sold_clean, field):
    """Shared bins over pooled 1st–99th percentiles; source data is unchanged."""
    title, unit = field_labels[field]
    datasets = {"Listings": finite_values(res_list_clean[field]),
                "Sold": finite_values(res_sold_clean[field])}
    combined = pd.concat(datasets.values())
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(datasets),
                        shared_xaxes="all", horizontal_spacing=0.2)
    if combined.empty:
        fig.add_annotation(text="No finite numeric values", showarrow=False)
        return fig
    lower, upper = combined.quantile([0.01, 0.99])
    if lower == upper:
        lower, upper = combined.min(), combined.max()
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    edges = np.linspace(lower, upper, 41)
    for col, (label, values) in enumerate(datasets.items(), 1):
        color = "#2563eb" if col == 1 else "#ea580c"
        visible = values[values.between(lower, upper)]
        counts, _ = np.histogram(visible, bins=edges)
        assert counts.sum() == len(visible)
        excluded = len(values) - len(visible)
        fig.layout.annotations[col - 1].text = f"{label} — {excluded:,} values omitted"
        ranges = [f"${a:,.0f} to ${b:,.0f}" if field in price_fields
                  else f"{a:,.1f} to {b:,.1f} {unit}"
                  for a, b in zip(edges[:-1], edges[1:])]
        trace = go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts,
                       width=np.diff(edges), customdata=ranges, name=label,
                       marker_color=color,
                       hovertemplate="%{customdata}<br>Count: %{y:,.0f}<extra>%{fullData.name}</extra>")
        fig.add_trace(trace, row=1, col=col)
        fig.update_xaxes(title_text=f"{title} ({unit})", range=[lower, upper],
                         tickformat="$,.0f" if field in price_fields else (",.2f" if field == "LotSizeAcres" else ",.0f"),
                         exponentformat="none", nticks=5, row=1, col=col)
    fig.update_yaxes(title_text="Number of records", tickformat=",.0f")
    fig.update_layout(title=f"{title} — pooled 1st–99th percentile range", template="plotly_white",
                      height=450, showlegend=False, dragmode="zoom", bargap=0.02)
    return fig



def save_histogram_png(histogram, field, output_path):
    """Save the interactive histogram's exact bins and counts as a static PNG."""
    title, unit = field_labels[field]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, annotation in zip(axes, histogram.layout.annotations[:2]):
        ax.set_title(annotation.text)
        ax.set_xlabel(f"{title} ({unit})")
        ax.set_ylabel("Number of records")
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    for ax, trace in zip(axes, histogram.data):
        ax.bar(trace.x, trace.y, width=np.asarray(trace.width) * 0.98,
               color=trace.marker.color)
    limits = histogram.layout.xaxis.range
    if limits is not None:
        ticks = np.linspace(limits[0], limits[1], 4)
        for ax in axes:
            ax.set_xlim(limits)
            ax.set_xticks(ticks)
            labels = [f"${value:,.0f}" if field in price_fields
                      else f"{value:,.2f}" if field == "LotSizeAcres"
                      else f"{value:,.0f}" for value in ticks]
            ax.set_xticklabels(labels, rotation=15, ha="right")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No finite numeric values", ha="center", transform=ax.transAxes)
    fig.suptitle(f"{title} — pooled 1st–99th percentile range")
    fig.tight_layout(w_pad=4)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def boxplot_figure(res_list_clean, res_sold_clean, field):
    """Notebook design: vertical boxes, price in millions, percentile axis zoom."""
    _, unit = field_labels[field]
    datasets = [finite_values(res_list_clean[field]), finite_values(res_sold_clean[field])]

    # Scale only the plotted values; saved data and summaries remain unchanged.
    if field in price_fields:
        datasets = [values / 1_000_000 for values in datasets]
        unit = "USD, millions"
    elif field == "LivingArea":
        unit = "square feet"

    combined = pd.concat(datasets)
    fig, ax = plt.subplots(figsize=(7, 5))
    if combined.empty:
        ax.set_title(f"{field}: no finite numeric values")
        return fig

    boxes = ax.boxplot(
        datasets,
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#111827", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.2},
    )
    for box, color in zip(boxes["boxes"], ["#93c5fd", "#fdba74"]):
        box.set_facecolor(color)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Listings", "Sold"])
    ax.set_ylabel(f"{field} ({unit})")
    ax.set_title(f"{field} distribution")

    # Zoom only: boxplot statistics still include all finite values.
    lower, upper = combined.quantile([0.01, 0.99])
    if lower < upper:
        padding = (upper - lower) * 0.05
        ax.set_ylim(lower - padding, upper + padding)

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5, 0.02,
        "View: pooled 1st–99th percentiles with padding; statistics use all values.",
        ha="center", fontsize=9, color="gray",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def main(show_plots=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_types = raw_property_types()
    raw_types.to_csv(OUTPUT_DIR / "raw_property_types.csv", index=False)
    cleaned = {}
    full_residential = {}
    sections = ["Week 2 Residential data report",
                "Week 1 filter: PropertyType == 'Residential'. Week 2 audits nulls first, "
                "drops BusinessType as out of scope for this Residential analysis, and selects the notebook's ten fields. "
                "All listing rows are retained. Sold EDA rows require a nonmissing ClosePrice. "
                "No duplicates or outliers are removed and no values are imputed.",
                "Available raw property types are audited separately; the Residential inputs have already "
                "been filtered by Week 1. Graphs and numeric summaries exclude nonnumeric and infinite values "
                "without changing the saved EDA data. Prices are USD; LivingArea uses square feet per the primer.",
                "Unique property types in available raw files", raw_types.to_string(index=False)]
    if raw_types.empty:
        sections.append("Raw files unavailable: original property types could not be verified.")
    for label, filename in [("Listings", "listings_residential.csv"), ("Sold", "sold_residential.csv")]:
        df = pd.read_csv(INPUT_DIR / filename, low_memory=False, dtype={"ListingId": "string"})
        input_types = df["PropertyType"].value_counts(dropna=False).to_frame("count")
        if not df["PropertyType"].eq("Residential").all():
            raise ValueError(f"{filename} contains non-Residential or missing PropertyType values; rerun Week 1")
        report = missing_val_check(df)
        flags = report[report["over_90_pct"]]
        report.to_csv(OUTPUT_DIR / f"{label.lower()}_missing.csv", index_label="field")
        flags.to_csv(OUTPUT_DIR / f"{label.lower()}_missing_over_90.csv", index_label="field")
        df = df.drop(columns=["BusinessType"], errors="ignore")
        full_residential[label] = df
        keep = df["ClosePrice"].notna() if label == "Sold" else pd.Series(True, index=df.index)
        clean = df.loc[keep, num_dist_field].copy()
        cleaned[label] = clean
        clean.to_csv(OUTPUT_DIR / f"{label.lower()}_residential_clean.csv", index=False)
        assert len(clean) == int(keep.sum()) and list(clean.columns) == num_dist_field
        numeric = clean[["ClosePrice", "LivingArea", "DaysOnMarket"]].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        summary = numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.rename(columns={"50%": "median"})
        summary.to_csv(OUTPUT_DIR / f"{label.lower()}_numeric_summary.csv", index_label="field")
        sections.extend([f"{label}", input_types.to_string(),
                         f"Rows before: {len(df):,}; rows after: {len(clean):,}; removed: {len(df)-len(clean):,}. "
                         f"Columns retained: {len(clean.columns)}. ListingId is an identifier, not a plotted numeric measure.",
                         "Columns above 90% null (flags, not automatic deletions)", flags.to_string(),
                         "Full null-count and percentage table", report.to_string(), "",
                         "Numeric distribution summary", summary.to_string(float_format=lambda v: f"{v:,.2f}")])
        print(f"{label}: {len(df):,} -> {len(clean):,} rows; {len(flags)} columns above 90% null")
    sold = full_residential["Sold"]
    dom = finite_values(sold["DaysOnMarket"])
    original = pd.to_numeric(sold["OriginalListPrice"], errors="coerce")
    close = pd.to_numeric(sold["ClosePrice"], errors="coerce")
    comparable = original.notna() & close.notna() & np.isfinite(original) & np.isfinite(close)
    higher = int(((original > close) & comparable).sum())
    denominator = int(comparable.sum())
    pct = f"{higher / denominator * 100:.2f}%" if denominator else "unavailable"
    counties = sold.assign(ClosePrice=close.replace([np.inf, -np.inf], np.nan)).groupby("CountyOrParish").agg(
        median_sold_price=("ClosePrice", "median"), sale_count=("ClosePrice", "count")
    ).sort_values("median_sold_price", ascending=False)
    counties.to_csv(OUTPUT_DIR / "county_median_prices.csv")
    sections.extend(["Notebook questions",
                     f"Sold Days on Market: mean {dom.mean():,.2f} days; median {dom.median():,.2f} days. "
                     f"Negative values: {(dom < 0).sum():,}. Mean versus median and the plots help assess skew.",
                     f"Original asking price exceeded sale price in {higher:,} of {denominator:,} records "
                     f"with both finite prices ({pct}). Missing prices are excluded from the denominator. "
                     "This differs from dividing by all sold rows in the scratch notebook.",
                     "Highest county median sale prices, entire available period",
                     "Counts are nonmissing sale prices, not deduplicated transactions. Small samples can produce unstable rankings.",
                     counties.head(10).to_string(float_format=lambda v: f"{v:,.2f}"),
                     "Histograms and boxplots\nHistograms show the pooled 1st–99th percentile range with omitted-value counts; CSVs and numeric summaries retain the full data. Boxplot statistics use all finite values; their axes show the pooled 1st–99th percentiles with 5% padding. Boxplot prices are displayed in USD millions. "
                     "Boxplot whiskers use 1.5 IQR; dots are review candidates, not confirmed errors. "
                     "Interactive histograms: drag to box zoom; double-click to reset. Scroll zoom is disabled. Both histograms and boxplots are also saved as static PNGs."])
    # Plain-text report contains the same tables and notebook answers.
    text = "\n\n".join(sections)
    (OUTPUT_DIR / "week2_report.txt").write_text(text, encoding="utf-8")
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    for field in num_dist_field[1:]:
        histogram = distribution_figure(cleaned["Listings"], cleaned["Sold"], field)
        save_histogram_png(histogram, field, plot_dir / f"{field}_histogram.png")
        if show_plots:
            histogram.show(config=PLOT_CONFIG)
        boxplot = boxplot_figure(cleaned["Listings"], cleaned["Sold"], field)
        boxplot.savefig(plot_dir / f"{field}_boxplot.png", dpi=150)
        if show_plots:
            plt.show()
        plt.close(boxplot)
    print(f"Reports and clean CSVs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Do not open interactive plots")
    args = parser.parse_args()
    main(show_plots=not args.no_plots)

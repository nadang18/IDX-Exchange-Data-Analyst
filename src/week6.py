"""Week 6: property features, school districts, and grouped/monthly EDA.

Run python src/week6.py. Inputs are Week 4 cleaned Residential CSVs and
2024–25 district polygons. Outputs go to data/processed/week6 (private).
Mortgage-rate enrichment is handled separately in Week 3.
Interactive monthly plots are for Jupyter; importing this module runs nothing.
Feature completeness filtering follows the notebook, including for listings:
rows missing any generated feature are removed. No deduplication is performed.
"""
from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data/processed/week4"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/week6"
DISTRICT_PATH = PROJECT_ROOT / "data/public/DistrictAreas2425/DistrictAreas2425.shp"
FEATURE_COLUMNS = ["PriceRatio", "PricePerSqft", "Month", "Year", "YrMo",
                   "ListtoContractDays", "ContracttoCloseDays"]
EXAMPLE_MEASURES = ["ClosePrice", "ListPrice", "PriceRatio", "DaysOnMarket"]


def show_table(table):
    """Show all rows/columns in Jupyter or print the full table in a terminal."""
    try:
        from IPython import get_ipython
        from IPython.display import display
        in_notebook = get_ipython() is not None and hasattr(get_ipython(), "kernel")
    except ImportError:
        in_notebook = False
    if in_notebook:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(table)
    else:
        print(table.to_string())


def load_districts(path=DISTRICT_PATH):
    return gpd.read_file(path)[
        ["CDCode", "DistrictNa", "DistrictTy", "Year", "geometry"]
    ].rename(columns={"DistrictNa": "DistrictName", "DistrictTy": "DistrictType"})


def add_property_features(df):
    featured = df.copy()

    
    date_fields = ["CloseDate", "PurchaseContractDate", "ListingContractDate"]
    featured[date_fields] = featured[date_fields].apply(
        pd.to_datetime, format="mixed", errors="raise"
    )

    featured = featured.assign(
        PriceRatio=featured["ClosePrice"] / featured["OriginalListPrice"],
        PricePerSqft=featured["ClosePrice"] / featured["LivingArea"],
        Month=featured["CloseDate"].dt.month,
        Year=featured["CloseDate"].dt.year,
        YrMo=featured["CloseDate"].dt.to_period("M"),
        ListtoContractDays=(
            featured["PurchaseContractDate"] - featured["ListingContractDate"]
        ).dt.days,
        ContracttoCloseDays=(
            featured["CloseDate"] - featured["PurchaseContractDate"]
        ).dt.days,
    )
    generated_columns = [
        "PriceRatio", "PricePerSqft", "Month", "Year", "YrMo",
        "ListtoContractDays", "ContracttoCloseDays",
    ]
    cleaned = featured.dropna(subset=generated_columns).copy()
    print(f"Before: {len(df):,}; removed: {len(df) - len(cleaned):,}; after: {len(cleaned):,}")
    return cleaned


def add_school_districts(df, districts):
    # Copy original df and check argument validity
    result = df.copy()
    boundaries = districts[
        ["CDCode", "DistrictName", "DistrictType", "Year", "geometry"]
    ].copy()

    if boundaries.crs is None:
        raise ValueError("District boundaries need a coordinate reference system.")

    if boundaries.empty or boundaries["Year"].nunique() != 1:
        raise ValueError("Provide district boundaries for one school year.")

    if boundaries[
        ["CDCode", "DistrictName", "DistrictType", "Year"]
    ].isna().any().any():
        raise ValueError("District attributes contain missing values.")

    boundaries["CDCode"] = boundaries["CDCode"].astype("string").str.zfill(7)
    boundaries["geometry"] = boundaries.geometry.make_valid()

    if boundaries.geometry.isna().any() or boundaries.geometry.is_empty.any():
        raise ValueError("District boundaries contain missing or empty geometry.")

    # Match each property to a temporary positional ID to identify it uniquely.

    annotations = pd.DataFrame(index=range(len(df)))

    lat = pd.to_numeric(df["Latitude"], errors="coerce").reset_index(drop=True)
    lon = pd.to_numeric(df["Longitude"], errors="coerce").reset_index(drop=True)

    missing = lat.isna() | lon.isna()

    # Global coordinate validity; district polygons determine coverage. 
    valid = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & lat.ne(0)
        & lon.ne(0)
    ).fillna(False)

    # Keep track of matching status based on coordinate
    annotations["SchoolDistrictMatchStatus"] = "unmatched"
    annotations.loc[
        missing, "SchoolDistrictMatchStatus"
    ] = "missing_or_nonnumeric_coordinates"
    annotations.loc[
        ~missing & ~valid, "SchoolDistrictMatchStatus"
    ] = "invalid_coordinates"

    # Turn usable coordinates into geographic points
    points = gpd.GeoDataFrame(
        {"property_row": annotations.index[valid]},
        geometry=gpd.points_from_xy(
            lon[valid].astype(float),
            lat[valid].astype(float),
        ),
        crs="EPSG:4326",
    )

    # Put property points and boundaries in the same CRS.
    points = points.to_crs(boundaries.crs)

    # Find district polygons containing or touching each property and match
    matches = gpd.sjoin(
        points,
        boundaries,
        how="inner",
        predicate="intersects",
    )

    # Multiple polygon pieces of the SAME district count as one match.
    matches = matches.drop_duplicates(
        subset=["property_row", "DistrictType", "CDCode"]
    )

    # Store each district type in separate columns.
    ambiguous = pd.Series(False, index=annotations.index)

    for district_type in ("Elementary", "High", "Unified"):
        subset = matches.loc[
            matches["DistrictType"] == district_type
        ].sort_values("CDCode")

        grouped = subset.groupby("property_row").agg(
            names=("DistrictName", lambda s: " | ".join(s.astype(str))),
            codes=("CDCode", lambda s: " | ".join(s.astype(str))),
        )

        annotations[f"SchoolDistrict{district_type}"] = (
            annotations.index.to_series().map(grouped["names"]).astype("string")
        )
        annotations[f"SchoolDistrict{district_type}Code"] = (
            annotations.index.to_series().map(grouped["codes"]).astype("string")
        )

        # Counts are used internally, not added to the returned DataFrame.
        counts = subset.groupby("property_row").size()
        ambiguous |= counts.reindex(annotations.index, fill_value=0).gt(1)

    # Check if property has multiple matches and mark as ambiguous
    matched_rows = matches["property_row"].unique()
    annotations.loc[matched_rows, "SchoolDistrictMatchStatus"] = "matched"
    annotations.loc[
        ambiguous, "SchoolDistrictMatchStatus"
    ] = "multiple_matches_same_type"

    annotations["SchoolDistrictAmbiguous"] = ambiguous
    annotations["SchoolDistrictBoundaryYear"] = str(boundaries["Year"].iloc[0])

    # Remove any old match-count columns from previous runs.
    old_count_columns = [
        column for column in result.columns
        if column.startswith("SchoolDistrict") and column.endswith("MatchCount")
    ]
    result = result.drop(columns=old_count_columns)

    # Attach by position, preserving the original index and row order.
    for column in annotations:
        result[column] = annotations[column].to_numpy()

    assert len(result) == len(df)
    assert result.index.equals(df.index)

    return result


def summarize_groups(df, measures, groupbys):
    if isinstance(measures, str):
        measures = [measures]
    if isinstance(groupbys, str):
        groupbys = [groupbys]

    def q25(values):
        return values.quantile(0.25)

    def q75(values):
        return values.quantile(0.75)

    def iqr(values):
        return values.quantile(0.75) - values.quantile(0.25)

    grouped = df.groupby(
        groupbys,
        dropna=False,   
        observed=True,
    )

    record_counts = grouped.size().rename("records")
    tables = {}

    for measure in measures:
        table = grouped[measure].agg(
            valid_count="count",
            mean="mean",
            median="median",
            std_dev="std",
            variance="var",
            minimum="min",
            percentile_25=q25,
            percentile_75=q75,
            maximum="max",
            IQR=iqr,
        )

        table.insert(0, "records", record_counts)
        table.insert(
            2,
            "missing_count",
            table["records"] - table["valid_count"],
        )

        tables[measure] = table.sort_values(
            "median", ascending=False
        )

    return tables


def plot_monthly_summary(df, measures, groupbys=None):
    """Display monthly plots and controls; return no summary tables."""

    import ipywidgets as widgets
    from IPython.display import display
    from matplotlib.figure import Figure
    from io import BytesIO
    import base64

    if isinstance(measures, str):
        measures = [measures]

    if groupbys is None:
        groupbys = []
    elif isinstance(groupbys, str):
        groupbys = [groupbys]

    # Keep one plot per distinct requested measure.
    measures = list(dict.fromkeys(measures))
    groupbys = list(dict.fromkeys(groupbys))

    if not measures:
        raise ValueError("Provide at least one measure.")
    if "YrMo" in groupbys:
        raise ValueError("YrMo is already the time axis.")

    required = ["YrMo", *measures, *groupbys]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Prepare a copy; the input DataFrame is unchanged.
    data = df[list(dict.fromkeys(required))].copy()
    data["YrMo"] = pd.to_datetime(
        data["YrMo"].astype("string"),
        format="%Y-%m",
        errors="coerce",
    ).dt.to_period("M")

    missing_time = data["YrMo"].isna().sum()
    data = data.loc[data["YrMo"].notna()].copy()

    if data.empty:
        raise ValueError("No valid YrMo values.")

    for measure in measures:
        data[measure] = pd.to_numeric(
            data[measure], errors="coerce"
        ).replace([float("inf"), float("-inf")], float("nan"))

    # Give each group combination an internal ID.
    if groupbys:
        data["_group_id"] = data.groupby(
            groupbys, dropna=False, observed=True, sort=False
        ).ngroup()

        group_labels = {}
        for _, row in data[
            ["_group_id", *groupbys]
        ].drop_duplicates("_group_id").iterrows():
            group_labels[row["_group_id"]] = " | ".join(
                f"{column}={row[column] if pd.notna(row[column]) else '(missing)'}"
                for column in groupbys
            )
    else:
        data["_group_id"] = 0
        group_labels = {0: "All records"}

    # Calculate once; selections only change what is plotted.
    summaries = {}

    for measure in measures:
        grouped = data.groupby(
            ["YrMo", "_group_id"], observed=True
        )[measure]

        table = grouped.agg(
            records="size",
            valid_count="count",
            mean="mean",
            median="median",
            std_dev="std",
            variance="var",
            minimum="min",
            percentile_25=lambda values: values.quantile(0.25),
            percentile_75=lambda values: values.quantile(0.75),
            maximum="max",
        )

        table["missing_count"] = table["records"] - table["valid_count"]
        table["IQR"] = table["percentile_75"] - table["percentile_25"]
        summaries[measure] = table.reset_index()

    months = pd.period_range(
        data["YrMo"].min(), data["YrMo"].max(), freq="M"
    )

    statistic = widgets.RadioButtons(
        options=[
            "mean", "median", "IQR", "std_dev", "variance",
            "minimum", "percentile_25", "percentile_75", "maximum",
            "records", "valid_count", "missing_count",
        ],
        value="median",
        description="Statistic:",
    )

    group_checks = [
        (
            group_id,
            widgets.Checkbox(
                value=not bool(groupbys),
                description=label,
                indent=False,
                layout=widgets.Layout(width="auto"),
            ),
        )
        for group_id, label in group_labels.items()
    ]

    output = widgets.Output()

    units = {
        "PriceRatio": "ratio",
        "ClosePrice": "USD",
        "ListPrice": "USD",
        "OriginalListPrice": "USD",
        "DaysOnMarket": "days",
        "LivingArea": "square feet",
        "LotSizeAcres": "acres",
        "PricePerSqft": "USD per square foot",
    }

    def draw(*_):
        rendered = []

        selected = [
            group_id for group_id, checkbox in group_checks
            if checkbox.value
        ]

        if not selected:
            output.outputs = ({"output_type": "stream", "name": "stdout",
                               "text": "Select at least one group.\n"},)
            return

        for measure in measures:
            table = summaries[measure]
            # Unmanaged figures cannot be redisplayed by pyplot.show().
            fig = Figure(figsize=(11, 4))
            ax = fig.subplots()

            for group_id in selected:
                series = (
                    table.loc[table["_group_id"] == group_id]
                    .set_index("YrMo")[statistic.value]
                    .reindex(months)
                )

                ax.plot(
                    months.to_timestamp(),
                    series.to_numpy(dtype=float, na_value=float("nan")),
                    marker="o",
                    markersize=3,
                    label=group_labels[group_id],
                )

            unit = units.get(measure, "")
            if statistic.value in {
                "records", "valid_count", "missing_count"
            }:
                unit = "records"
            elif statistic.value == "variance" and unit:
                unit = f"{unit}²"

            ax.set_title(f"{measure} — monthly {statistic.value}")
            ax.set_xlabel("Year–month")
            ax.set_ylabel(
                f"{statistic.value}"
                + (f" ({unit})" if unit else "")
            )
            ax.ticklabel_format(
                axis="y", style="plain", useOffset=False
            )
            ax.grid(alpha=0.25)

            if groupbys:
                ax.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )

            fig.autofmt_xdate()
            fig.tight_layout()
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            rendered.append({
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
                "metadata": {},
            })

        # Replace the entire widget output once, rather than append displays.
        output.outputs = tuple(rendered)


    statistic.observe(draw, names="value")
    for _, checkbox in group_checks:
        checkbox.observe(draw, names="value")

    controls = [statistic]
    if groupbys:
        controls.extend([
            widgets.HTML("<b>Groups to display:</b>"),
            widgets.VBox(
                [checkbox for _, checkbox in group_checks],
                layout=widgets.Layout(
                    max_height="220px", overflow="auto"
                ),
            ),
        ])

    print(f"Rows excluded for missing/invalid YrMo: {missing_time:,}")
    display(widgets.VBox([*controls, output]))
    draw()

def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, show_plots=False,
         sample_rows=10):
    """Save outputs and return featured frames, summaries, and audit tables.

    Set output_dir to a test folder to avoid overwriting regular Week 6 outputs.
    Set show_plots=True in Jupyter to display the interactive county examples.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    districts = load_districts()
    show_table(districts.head(sample_rows))
    show_table(districts["DistrictType"].value_counts())

    featured_frames, summaries, counts = {}, {}, []
    for label in ["Listings", "Sold"]:
        source = pd.read_csv(
            input_dir / f"{label.lower()}_residential_clean.csv",
            dtype={"ListingKey": "string", "ListingId": "string", "PostalCode": "string"},
            low_memory=False,
        )
        required = ["PropertySubType", "MLSAreaMajor", "ListOfficeName", "BuyerOfficeName"]
        missing = [c for c in required if c not in source]
        if missing:
            raise ValueError(f"{label}: missing {missing}; rerun the updated Week 4 script.")

        print(f"\n{label}: property features")
        featured = add_property_features(source)
        counts.append({"dataset": label, "before": len(source),
                       "removed": len(source) - len(featured), "after": len(featured)})
        show_table(featured[FEATURE_COLUMNS].head(sample_rows))
        featured[FEATURE_COLUMNS].head(sample_rows).to_csv(
            output_dir / f"{label.lower()}_feature_sample.csv", index=False)
        nulls = pd.DataFrame({"null_count": featured.isna().sum(),
                              "null_percent": featured.isna().mean().mul(100)})
        show_table(nulls)
        nulls.to_csv(output_dir / f"{label.lower()}_feature_nulls.csv", index_label="column")

        before_districts = len(featured)
        featured = add_school_districts(featured, districts)
        assert len(featured) == before_districts
        featured_frames[label] = featured
        featured.to_csv(output_dir / f"{label.lower()}_residential_featured.csv", index=False)

        print(label, "rows:", len(featured))
        columns = ["ListingId", "Latitude", "Longitude", "SchoolDistrictElementary",
                   "SchoolDistrictHigh", "SchoolDistrictUnified", "SchoolDistrictBoundaryYear",
                   "SchoolDistrictMatchStatus"]
        show_table(featured[columns].head(sample_rows))
        review = featured.loc[featured["SchoolDistrictMatchStatus"] != "matched", columns]
        print(f"Records requiring match review (including ambiguous): {len(review):,}")
        show_table(review.head(sample_rows))
        review.head(sample_rows).to_csv(output_dir / f"{label.lower()}_district_review_sample.csv", index=False)

        summaries[label] = summarize_groups(featured, EXAMPLE_MEASURES, "CountyOrParish")
        for measure, table in summaries[label].items():
            print(f"\n{label}: {measure} by CountyOrParish")
            show_table(table)
            table.to_csv(output_dir / f"{label.lower()}_county_{measure}_summary.csv")
        if show_plots:
            plot_monthly_summary(featured, EXAMPLE_MEASURES, "CountyOrParish")

    match_summary = pd.DataFrame({
        label: frame["SchoolDistrictMatchStatus"].value_counts()
        for label, frame in featured_frames.items()
    }).fillna(0).astype(int)
    show_table(match_summary)
    match_summary.to_csv(output_dir / "district_match_summary.csv", index_label="status")
    row_counts = pd.DataFrame(counts)
    show_table(row_counts)
    row_counts.to_csv(output_dir / "feature_row_counts.csv", index=False)
    (output_dir / "week6_notes.txt").write_text(
        "Features follow the notebook. PriceRatio = ClosePrice / OriginalListPrice; "
        "PricePerSqft = ClosePrice / LivingArea. Year, Month, YrMo use CloseDate. "
        "ContracttoCloseDays = CloseDate - PurchaseContractDate. "
        "Rows missing any generated feature are removed, including listings without closing dates. "
        "Missing metadata alone does not remove rows. No deduplication. "
        "District assignment preserves every featured row, including missing coordinates. "
        "2024–25 boundaries apply to every sale year. Multiple matches of the same type "
        "are retained and flagged; match counts are internal only. "
        "Monthly plots use YrMo and start with no county selected. "
        "Feature null reports are after feature filtering and before district enrichment. "
        "Mortgage enrichment is handled separately in Week 3.\n", encoding="utf-8")
    print(f"Saved Week 6 outputs to {output_dir}")
    return {"listings": featured_frames["Listings"], "sold": featured_frames["Sold"],
            "summaries": summaries, "row_counts": row_counts,
            "district_match_summary": match_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)

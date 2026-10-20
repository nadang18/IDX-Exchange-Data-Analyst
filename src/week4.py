"""Weeks 4–5: reproducible Residential data preparation from the Week 4 notebook.

Run: python src/week4.py (pandas >= 2 required).
Input: data/processed/week1/{listings,sold}_residential.csv.
Output: data/processed/week4/. No API requests or mortgage-rate merge.

Workflow: parse dates -> numeric flags/review/remove/recheck -> date flags/
review/remove/recheck -> geographic flags/review/remove/recheck -> select
notebook columns -> confirm numeric/text/date types -> export CSV and audits.

Missing dates and coordinates are retained. Same-day dates are allowed.
Zero coordinates ARE removed, matching the current notebook. Geographic bounds
are a coarse California screen, not an exact state polygon. No imputation,
statistical outlier trimming, or deduplication is performed.
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week4"
DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
CA_LAT_BOUNDS = (32.0, 42.1)
CA_LON_BOUNDS = (-124.6, -114.0)
NUMERIC_TYPES = {
    "OriginalListPrice": "Float64", "ListPrice": "Float64", "ClosePrice": "Float64",
    "LivingArea": "Float64", "Latitude": "Float64", "Longitude": "Float64",
    "DaysOnMarket": "Int64", "YearBuilt": "Int64", "BedroomsTotal": "Int64",
    "BathroomsTotalInteger": "Int64",
}

fin_metadata = [
    "ListingKey",
    "ListingId",
    "OriginalListPrice",
    "ListPrice",
    "ClosePrice",
    "ListingContractDate",
    "PurchaseContractDate",
    "CloseDate",
    "ContractStatusChangeDate",
    "DaysOnMarket",
    "MlsStatus",
    "UnparsedAddress",
    "City",
    "CountyOrParish",
    "StateOrProvince",
    "PostalCode",
    "Latitude",
    "Longitude",
    "YearBuilt",
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
]

def remove_invalid_rows(df, flags, flag_column):
    if not df.index.equals(flags.index):
        raise ValueError("Data and flag indexes do not match.")

    # Missing flags are treated as False, so those rows are retained.
    remove = flags[flag_column].fillna(False)
    cleaned = df.loc[~remove].copy()

    print(f"Before: {len(df):,}")
    print(f"Removed: {len(df) - len(cleaned):,}")
    print(f"After: {len(cleaned):,}")

    return cleaned


def flag_invalid_values(df):
    flags = pd.DataFrame(index=df.index)

    for field in ["ClosePrice", "LivingArea"]:
        values = pd.to_numeric(df[field], errors="coerce")
        flags[f"{field}_invalid"] = (values <= 0).fillna(False)

    for field in ["DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger"]:
        values = pd.to_numeric(df[field], errors="coerce")
        flags[f"{field}_invalid"] = (values < 0).fillna(False)

    flags["any_invalid_numeric"] = flags.any(axis=1)
    return flags


def flag_date_consistency(df):
    flags = pd.DataFrame(index=df.index)
    date_columns = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
    dates = df[date_columns].apply(
        pd.to_datetime, format="mixed", errors="coerce"
    )
    listing = dates["ListingContractDate"]
    purchase = dates["PurchaseContractDate"]
    close = dates["CloseDate"]

    flags["listing_after_close_flag"] = (
        listing.notna() & close.notna() & (listing > close)
    ).fillna(False)

    flags["purchase_after_close_flag"] = (
        purchase.notna() & close.notna() & (purchase > close)
    ).fillna(False)

    listing_after_purchase = (
        listing.notna() & purchase.notna() & (listing > purchase)
    ).fillna(False)
    
    # Overall flag: any observed part of listing -> purchase -> close is reversed/out of order.
    flags["negative_timeline_flag"] = (
        listing_after_purchase
        | flags["listing_after_close_flag"]
        | flags["purchase_after_close_flag"]
    )
    return flags


def flag_geographic_values(df):
    flags = pd.DataFrame(index=df.index)
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")

    flags["missing_coordinates_flag"] = (
        df["Latitude"].isna() | df["Longitude"].isna()
    )

    flags["nonnumeric_coordinates_flag"] = (
        (df["Latitude"].notna() & lat.isna())
        | (df["Longitude"].notna() & lon.isna())
    )

    flags["zero_coordinates_flag"] = (lat.eq(0) | lon.eq(0)).fillna(False)

    flags["positive_longitude_flag"] = lon.gt(0).fillna(False)

    flags["implausible_coordinates_flag"] = (
        (lat.notna() & ~lat.between(-90, 90))
        | (lon.notna() & ~lon.between(-180, 180))
    ).fillna(False)


    complete = lat.notna() & lon.notna() & lat.ne(0) & lon.ne(0)

    flags["outside_ca_bounds_flag"] = (
        complete
        & ~(lat.between(*CA_LAT_BOUNDS) & lon.between(*CA_LON_BOUNDS))
    ).fillna(False)

    flags["any_geographic_issue"] = flags.any(axis=1)

    #Choose remove condition
    flags["remove_geographic_flag"] = flags[[
        "positive_longitude_flag",
        "zero_coordinates_flag",
        "implausible_coordinates_flag",
        "outside_ca_bounds_flag",
    ]].any(axis=1)
    return flags

TRANSFORMATIONS = [
    ("input", "Use Week 1 Residential inputs; verify PropertyType is Residential. Keep all listing statuses and do not require ClosePrice."),
    ("date_conversion", "Parse four date fields with mixed formats; unparseable values become NaT and are counted. Missing dates alone do not remove rows."),
    ("numeric", "Remove ClosePrice <= 0 or LivingArea <= 0; remove negative DaysOnMarket, BedroomsTotal, or BathroomsTotalInteger. Missing values do not trigger these range rules."),
    ("dates", "Remove reversed observed date pairs, including listing after purchase. Allow equal dates; compare only present dates."),
    ("geography", "Remove zero coordinates, positive longitude, values outside global latitude/longitude ranges, or complete nonzero pairs outside latitude 32.0–42.1 / longitude -124.6–-114.0. Retain missing coordinates. Nonnumeric coordinates are flagged and require review at the typing step."),
    ("column_selection", "Keep exactly fin_metadata from the notebook. This omits BusinessType and repeated export columns; does not drop duplicate records."),
    ("type_confirmation", "Use nullable Float64 for decimal measures and Int64 for whole-number measures. Preserve text identifiers and missing values. Stop on unparseable numeric strings, infinities, or fractional integer values rather than silently rounding or filling."),
    ("export", "Write dates in ISO form and omit the pandas index. CSV does not preserve pandas types; use the saved schema when reloading."),
]


def standardize_types(df):
    """Enforce consistent final types without silently changing questionable numbers."""
    result = df.copy()
    for field, dtype in NUMERIC_TYPES.items():
        values = pd.to_numeric(result[field], errors="raise")
        if values.isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"{field}: infinite values require review")
        if dtype == "Int64" and (values.dropna() % 1 != 0).any():
            raise ValueError(f"{field}: fractional values require review before Int64 conversion")
        result[field] = values.astype(dtype)
    for field in result.columns:
        if field not in NUMERIC_TYPES and field not in DATE_FIELDS:
            result[field] = result[field].astype("string")
    return result


def process_dataset(label, input_path, output_dir):
    """Run each notebook stage with independent flags and auditable removal counts."""
    source = pd.read_csv(input_path, low_memory=False,
                         dtype={"ListingKey": "string", "ListingId": "string", "PostalCode": "string"})
    if not source["PropertyType"].eq("Residential").all():
        raise ValueError(f"{label}: input must contain only Residential records; rerun Week 1")
    df = source.copy()
    counts = []
    flag_counts = []
    parse_counts = []
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    def count_stage(stage, before, after):
        counts.append({"dataset": label, "stage": stage, "before": before,
                       "removed": before-after, "after": after})

    count_stage("input", len(df), len(df))
    for field in DATE_FIELDS:
        original = df[field]
        converted = pd.to_datetime(original, format="mixed", errors="coerce")
        parse_counts.append({"dataset": label, "field": field,
                             "missing_before": int(original.isna().sum()),
                             "unparseable_to_NaT": int((original.notna() & converted.isna()).sum()),
                             "missing_after": int(converted.isna().sum())})
        df[field] = converted
    count_stage("date_conversion", len(df), len(df))

    stages = [
        ("numeric", flag_invalid_values, "any_invalid_numeric"),
        ("dates", flag_date_consistency, "negative_timeline_flag"),
        ("geography", flag_geographic_values, "remove_geographic_flag"),
    ]
    for stage, checker, removal_flag in stages:
        flags = checker(df)
        # Review/audit stage: preserve source-row identifiers, values, and flags
        # for every flagged record, including geographic issues we retain.
        issue = flags.any(axis=1)
        audit = df.loc[issue, fin_metadata].copy()
        audit.insert(0, "source_row_index", df.index[issue])
        for field in flags.columns:
            audit[field] = flags.loc[issue, field]
        audit.to_csv(audit_dir / f"{label}_{stage}_flagged.csv", index=False)
        before = len(df)
        print(f"\n{label}: {stage}")
        print(flags.sum().to_string())
        df = remove_invalid_rows(df, flags, removal_flag)
        count_stage(stage, before, len(df))
        rechecked = checker(df)
        if rechecked[removal_flag].any():
            raise ValueError(f"{label}: {stage} removal failed validation")
        for field in flags.columns:
            flag_counts.append({"dataset": label, "stage": stage, "flag": field,
                                "before": int(flags[field].sum()),
                                "after": int(rechecked[field].sum())})

    formatted = standardize_types(df.loc[:, fin_metadata].copy())
    count_stage("column_selection_and_types", len(df), len(formatted))
    for _, checker, removal_flag in stages:
        if checker(formatted)[removal_flag].any():
            raise ValueError(f"{label}: final validation failed for {removal_flag}")
    confirmations = pd.DataFrame([
        {"dataset": label, "field": field, "input_dtype": str(source[field].dtype),
         "final_dtype": str(formatted[field].dtype), "null_count": int(formatted[field].isna().sum())}
        for field in fin_metadata
    ])
    expected = {field: NUMERIC_TYPES.get(field, "datetime64[ns]" if field in DATE_FIELDS else "string")
                for field in fin_metadata}
    if any(str(formatted[field].dtype) != dtype for field, dtype in expected.items()):
        raise ValueError(f"{label}: final schema mismatch")
    schema = {"numeric_types": NUMERIC_TYPES, "date_fields": DATE_FIELDS,
              "string_fields": [c for c in fin_metadata if c not in NUMERIC_TYPES and c not in DATE_FIELDS]}
    (output_dir / f"{label}_schema.json").write_text(json.dumps(schema, indent=2)+"\n")
    formatted.to_csv(output_dir / f"{label}_residential_clean.csv", index=False,
                     date_format="%Y-%m-%dT%H:%M:%S")
    return pd.DataFrame(counts), pd.DataFrame(flag_counts), pd.DataFrame(parse_counts), confirmations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [process_dataset(label, INPUT_DIR / f"{label}_residential.csv", OUTPUT_DIR)
               for label in ["listings", "sold"]]
    counts, flags, dates, types = [pd.concat([r[i] for r in results], ignore_index=True) for i in range(4)]
    counts.to_csv(OUTPUT_DIR / "row_counts.csv", index=False)
    flags.to_csv(OUTPUT_DIR / "flag_counts.csv", index=False)
    flags.loc[flags["stage"] == "dates"].to_csv(OUTPUT_DIR / "date_consistency_summary.csv", index=False)
    flags.loc[flags["stage"] == "geography"].to_csv(OUTPUT_DIR / "geographic_quality_summary.csv", index=False)
    dates.to_csv(OUTPUT_DIR / "date_conversion_summary.csv", index=False)
    types.to_csv(OUTPUT_DIR / "dtype_confirmations.csv", index=False)
    report = ["Weeks 4–5 data preparation", "Transformations and reasons:"]
    report.extend(f"{step}: {reason}" for step, reason in TRANSFORMATIONS)
    report.extend(["Row counts (sequential stages; flag counts can overlap):", counts.to_string(index=False),
                   "Flag counts before removal and after rechecking:", flags.to_string(index=False),
                   "Date parsing and retained missing dates:", dates.to_string(index=False),
                   "Data type confirmations and final null counts:", types.to_string(index=False),
                   "Geographic limitations: the screening rectangle is not an exact state boundary. Missing coordinates remain, so not every retained row is suitable for mapping.",
                   "No mortgage rates merged. No duplicate-record removal or statistical outlier trimming. Missing values are not imputed.",
                   "Audit CSVs identify flagged records by source_row_index (zero-based position in the Week 1 input), ListingKey, and ListingId."])
    (OUTPUT_DIR / "week4_report.txt").write_text("\n\n".join(report)+"\n", encoding="utf-8")
    print(f"\nSaved cleaned datasets and documentation to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

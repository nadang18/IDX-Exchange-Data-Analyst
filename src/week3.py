"""Week 3: merge FRED monthly mortgage rates onto combined MLS datasets.

Run: python src/week3.py (requires pandas and an internet connection).
Matches the notebook's full sold_dfs/listing_dfs, not the Residential subsets.
Inputs: idx_data/csv/CRMLSListingYYYYMM.csv and CRMLSSoldYYYYMM[_filled].csv.
Outputs: data/processed/week3/. No rows are deduplicated or filtered by property type.

Source: Freddie Mac, MORTGAGE30US, retrieved from FRED, St. Louis Fed:
https://fred.stlouisfed.org/series/MORTGAGE30US
Rates are percentages (6.5 means 6.5%, not 0.065). Monthly rates are arithmetic
means of weekly observations dated within each calendar month, not daily-weighted
averages. No forward filling or interpolation is performed.
"""

from io import BytesIO
from pathlib import Path
import re
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "idx_data" / "csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week3"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
RATE_COLUMN = "rate_30yr_fixed"


def monthly_rates(mortgage):
    """Resample the weekly FRED series into one observation per calendar month."""
    mortgage = mortgage.copy()
    mortgage["observation_date"] = pd.to_datetime(mortgage["observation_date"], errors="raise")
    mortgage["MORTGAGE30US"] = pd.to_numeric(mortgage["MORTGAGE30US"], errors="raise")
    if mortgage.empty or mortgage["observation_date"].isna().any():
        raise ValueError("FRED returned no observations or missing observation dates")
    if mortgage["observation_date"].duplicated().any():
        raise ValueError("FRED returned duplicate observation dates")
    rates = (
        mortgage.set_index("observation_date")["MORTGAGE30US"]
        .sort_index().resample("MS").mean().rename(RATE_COLUMN).reset_index()
    )
    rates["year_month"] = rates["observation_date"].dt.to_period("M").astype(str)
    return rates[["year_month", RATE_COLUMN]]


def fetch_mortgage_rates():
    """Fetch directly from FRED's public CSV endpoint; no API key required."""
    with urlopen(FRED_URL, timeout=60) as response:
        mortgage = pd.read_csv(BytesIO(response.read()), na_values=["."])
    return monthly_rates(mortgage)


def load_combined(prefix):
    """Recreate the notebook's combined full dataset from monthly source files."""
    pattern = re.compile(rf"{prefix}(\d{{6}})(?:_filled)?\.csv")
    files = [p for p in sorted(DATA_DIR.glob(f"{prefix}*.csv")) if pattern.fullmatch(p.name)]
    if not files:
        raise FileNotFoundError(f"No monthly {prefix} files in {DATA_DIR}")
    months = [pattern.fullmatch(p.name).group(1) for p in files]
    if len(months) != len(set(months)):
        raise ValueError(f"Multiple {prefix} files for the same month; select one version first")
    frames = [pd.read_csv(p, low_memory=False, dtype={"ListingId": "string", "ListingKey": "string"}) for p in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"{prefix}: {len(files)} monthly files, {len(combined):,} rows")
    return combined


def merge_rates(df, rates, date_column):
    """Left merge preserves MLS rows; many-to-one validation prevents multiplication."""
    if "year_month" in df.columns or RATE_COLUMN in df.columns:
        raise ValueError("Input already contains enrichment columns; use the original combined data")
    dated = df.copy()
    dates = pd.to_datetime(dated[date_column], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError(f"{date_column}: {dates.isna().sum():,} missing/invalid dates; cannot assign a monthly rate")
    dated["year_month"] = dates.dt.to_period("M").astype(str)
    enriched = dated.merge(rates, on="year_month", how="left", validate="many_to_one", sort=False)
    if len(enriched) != len(df):
        raise ValueError("Merge changed the source row count")
    missing = enriched[RATE_COLUMN].isna()
    if missing.any():
        months = sorted(enriched.loc[missing, "year_month"].unique())
        raise ValueError(f"{missing.sum():,} rows have no mortgage rate; unmatched months: {months}")
    print(f"{date_column}: {len(enriched):,} rows after merge; null rates: 0 (PASS)")
    return enriched


def main():
    rates = fetch_mortgage_rates()
    sold_with_rates = merge_rates(load_combined("CRMLSSold"), rates, "CloseDate")
    listings_with_rates = merge_rates(load_combined("CRMLSListing"), rates, "ListingContractDate")

    # Validate both datasets before writing either enriched output.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sold_with_rates.to_csv(OUTPUT_DIR / "sold_with_rates.csv", index=False)
    listings_with_rates.to_csv(OUTPUT_DIR / "listings_with_rates.csv", index=False)
    rates.to_csv(OUTPUT_DIR / "mortgage_monthly.csv", index=False)
    validation = pd.DataFrame([
        {"dataset": label, "date_key": date_key, "rows": len(df),
         "null_rates": int(df[RATE_COLUMN].isna().sum()),
         "first_month": df["year_month"].min(), "last_month": df["year_month"].max()}
        for label, date_key, df in [
            ("Sold", "CloseDate", sold_with_rates),
            ("Listings", "ListingContractDate", listings_with_rates),
        ]
    ])
    validation.to_csv(OUTPUT_DIR / "merge_validation.csv", index=False)
    print(f"Saved enriched datasets, monthly rates, and validation to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

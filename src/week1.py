"""Combine available monthly CRMLS exports and keep Residential records only."""

from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd


# Get directories for input data CSVs and create output for processed files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "idx_data" / "csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
START_MONTH = "202401" # Starting of data range
END_MONTH = (date.today().replace(day=1) - timedelta(days=1)).strftime("%Y%m") # Get today's month in YYYYMM


# Listings: 860,898 rows 
# Listings(concat): 860,898 rows
# Listings(residential): 547,162 rows

# Sold: 615,707 source rows
# Sold(concat): 615,707 rows
# Sold(residential): 414,054 rows



def combine_residential(prefix, output_name):
    """Read one dataset's monthly files, concatenate, filter, and save."""
    frames = []
    months = []
    pattern = re.compile(rf"{prefix}(\d{{6}})(?:_filled)?\.csv")

    for file in sorted(DATA_DIR.glob(f"{prefix}*.csv")):
        match = pattern.fullmatch(file.name)
        if not match:
            continue

        month = match.group(1)
        if not START_MONTH <= month <= END_MONTH:
            continue

        df = pd.read_csv(file, low_memory=False)
        print(f"{file.name}: {len(df):,} rows before concatenation")
        frames.append(df)
        months.append(month)

    if not frames:
        raise FileNotFoundError(f"No matching monthly files in {DATA_DIR}")

    # Check length before and after concatenation
    before_concat = sum(len(df) for df in frames)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    after_concat = len(combined)
    assert before_concat == after_concat, "Concatenation changed the row count"

    # Residential only filter
    before_filter = len(combined)
    residential = combined.loc[combined["PropertyType"] == "Residential"]
    after_filter = len(residential)

    print(f"\n{prefix}: {len(frames)} files, {min(months)} through {max(months)}")
    print(f"Concatenation: {before_concat:,} before -> {after_concat:,} after")
    print(f"Residential filter: {before_filter:,} before -> {after_filter:,} after")

    # Separate outputs from inputs so they cannot be loaded on the next run.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    residential.to_csv(output_path, index=False)
    print(f"Saved {output_path}\n")


def main():
    combine_residential("CRMLSListing", "listings_residential.csv")
    combine_residential("CRMLSSold", "sold_residential.csv")


if __name__ == "__main__":
    main()

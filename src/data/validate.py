"""Data validation module - loads raw student datasets, checks for required columns,
and handles missing values before preprocessing.
"""
import pandas as pd
from src.common.utils import load_yaml, project_path

# Define columns that should be numeric (grades and attendance metrics)
EXPECTED_NUMERIC = {"age","absences","G1","G2","G3"}

def main():
    """Main validation pipeline:
    1. Load configuration and paths
    2. Read raw CSV files (student-mat.csv, student-por.csv)
    3. Merge datasets with source tracking
    4. Convert expected columns to numeric (coerce errors to NaN)
    5. Drop rows with missing G1, G2, or G3 values
    6. Save cleaned data to interim directory
    """
    # Load configuration from experiment_base.yaml
    cfg = load_yaml("config/experiment_base.yaml")
    raw_dir = project_path(cfg["paths"]["raw"])
    interim = project_path(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    # Load both mathematics and Portuguese course datasets from raw directory
    mat = pd.read_csv(raw_dir / "student-mat.csv", sep=";")
    por = pd.read_csv(raw_dir / "student-por.csv", sep=";")
    # Add dataset source identifier to each dataset
    mat["dataset"] = "mat"; por["dataset"] = "por"
    # Combine both datasets into single DataFrame
    df = pd.concat([mat, por], ignore_index=True)

    # Convert numeric columns, handling any non-numeric values by setting to NaN
    for c in EXPECTED_NUMERIC:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    # Remove rows with missing grade values (G1, G2, G3 are required for target creation)
    df = df.dropna(subset=["G1","G2","G3"])

    # Export cleaned data to interim directory for next preprocessing step
    df.to_csv(interim / "cleaned.csv", sep=";", index=False)
    print("[validate] cleaned.csv written")

if __name__ == "__main__":
    main()

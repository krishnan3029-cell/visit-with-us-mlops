
"""Validate the registered tourism dataset before downstream pipeline jobs."""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
}


def validate_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.columns.duplicated().any():
        raise ValueError("Dataset has duplicate column names.")

    if df["ProdTaken"].isna().any() or not df["ProdTaken"].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only 0 and 1, with no missing values.")

    if df["CustomerID"].isna().any() or df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique for each row.")

    print("DATA REGISTRATION / VALIDATION REPORT")
    print("=" * 42)
    print(f"Dataset: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Required columns verified: {len(EXPECTED_COLUMNS)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts().sort_index().to_string())

    print("\nTarget distribution (%):")
    print(
        (df["ProdTaken"].value_counts(normalize=True).sort_index() * 100)
        .round(2).to_string()
    )

    extra_columns = sorted(set(df.columns) - EXPECTED_COLUMNS)
    print(f"\nExtra columns (allowed, review during preparation): {extra_columns}")
    print("\nSUCCESS: Dataset validation passed.")

    return df


if __name__ == "__main__":
    try:
        validate_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: Dataset validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

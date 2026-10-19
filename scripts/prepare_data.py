
"""
Step 2: Data Preparation
Visit with Us - Tourism Purchase Prediction
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

print("Loading tourism dataset...")
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 3. Remove unnecessary and leakage columns
# --------------------------------------------------

# Customer identifiers and post-contact information
# must not be used for pre-contact predictions.

columns_to_drop = [
    "Unnamed: 0",
    "CustomerID",
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]

df = df.drop(columns=columns_to_drop, errors="ignore")

print("Shape after removing columns:", df.shape)

# --------------------------------------------------
# 4. Standardize categorical values
# --------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()

# Correct inconsistent labels
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

# --------------------------------------------------
# 5. Remove duplicate records
# --------------------------------------------------

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

# --------------------------------------------------
# 6. Validate the target variable
# --------------------------------------------------

if df["ProdTaken"].isna().any():
    raise ValueError("Missing values found in target column.")

if not df["ProdTaken"].isin([0, 1]).all():
    raise ValueError("ProdTaken must contain only 0 and 1.")

# --------------------------------------------------
# 7. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isna().sum())

# Missing feature values will be imputed inside
# the model pipeline in Step 3.
# This prevents leakage from the test dataset.

# --------------------------------------------------
# 8. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)

# --------------------------------------------------
# 9. Split into training and testing datasets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test

# --------------------------------------------------
# 10. Save processed datasets
# --------------------------------------------------

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# --------------------------------------------------
# 11. Print summary
# --------------------------------------------------

print("\nDATA PREPARATION COMPLETED")
print("-" * 40)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

print("\nSaved files:")
print(train_path)
print(test_path)

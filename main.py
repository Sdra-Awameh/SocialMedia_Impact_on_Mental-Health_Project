# Teen Mental Health - Exploratory Data Analysis (EDA)
# This script explores the dataset. It does NOT train a machine learning model.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Teen_Mental_Health_Dataset.csv"
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("TEEN MENTAL HEALTH DATASET")
print("=" * 60)

# --------------------------------------------------
# 2. Basic overview
# --------------------------------------------------
print("\nDataset shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 3. Target variable: depression_risk
# --------------------------------------------------
target = "depression_risk"
print("\nDepression risk class counts:")
print(df[target].value_counts())

print("\nDepression risk class percentages:")
print((df[target].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# 4. Descriptive statistics
# --------------------------------------------------
print("\nNumerical descriptive statistics:")
print(df.describe())

print("\nCategorical descriptive statistics:")
print(df.describe(include="object"))

# --------------------------------------------------
# 5. Visualizations
# --------------------------------------------------
sns.set_theme(style="whitegrid")

# Distribution of depression risk (low, medium, high)
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target, order=["low", "medium", "high"])
plt.title("Distribution of Depression Risk")
plt.xlabel("Depression Risk")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# Distribution of numerical columns
numeric_columns = df.select_dtypes(include="number").columns.tolist()
df[numeric_columns].hist(figsize=(14, 10), bins=15, edgecolor="black")
plt.suptitle("Distributions of Numerical Features", y=1.02)
plt.tight_layout()
plt.show()

# Compare selected variables across depression risk groups
comparison_columns = [
    "daily_social_media_hours",
    "sleep_hours",
    "stress_level",
    "anxiety_level",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
]
available_comparison_columns = [c for c in comparison_columns if c in df.columns]

for column in available_comparison_columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=target, y=column, order=["low", "medium", "high"])
    plt.title(f"{column.replace('_', ' ').title()} by Depression Risk")
    plt.xlabel("Depression Risk")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()
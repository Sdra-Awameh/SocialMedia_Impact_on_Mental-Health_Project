
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned dataset
df = pd.read_csv("Teen_Mental_Health_Dataset_Cleaned.csv")

# 2. Check dataset dimensions
print("Dataset shape:", df.shape)

# 3. Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 4. Display column names and data types
print("\n" + "=" * 50)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 50)

print(df.dtypes)

# 5. Check missing values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())

# 6. Check duplicate rows
print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)

print("Number of duplicate rows:", df.duplicated().sum())

# 7. Display descriptive statistics
print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

print(df.describe())

# 8. Target variable distribution
print("\n" + "=" * 50)
print("DEPRESSION LABEL DISTRIBUTION")
print("=" * 50)

print(df["depression_label"].value_counts().sort_index())

print("\nPercentages:")
print(
    (df["depression_label"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

# Plot the target distribution
sns.countplot(data=df, x="depression_label")

plt.title("Distribution of Depression Label")
plt.xlabel("Depression Label")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# 9. Distribution of daily social media usage

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="daily_social_media_hours",
    bins=10,
    kde=True
)

plt.title("Distribution of Daily Social Media Usage")
plt.xlabel("Daily Social Media Hours")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# 10. Correlation analysis

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 11. Categorical feature distributions

categorical_columns = [
    "gender",
    "platform_usage",
    "social_interaction_level"
]

for column in categorical_columns:
    print(f"\nDistribution of {column}:")
    print(df[column].value_counts())

    plt.figure(figsize=(7, 4))

    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Number of Students")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


# 12. Compare numerical features by depression label

features_to_compare = [
    "daily_social_media_hours",
    "sleep_hours",
    "stress_level",
    "anxiety_level"
]

for column in features_to_compare:
    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=df,
        x="depression_label",
        y=column
    )

    plt.title(f"{column.replace('_', ' ').title()} by Depression Label")
    plt.xlabel("Depression Label")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()


# Correlation with depression_label
print("\n" + "=" * 50)
print("CORRELATION WITH DEPRESSION LABEL")
print("=" * 50)

numeric_df = df.select_dtypes(include="number")
correlation_with_target = numeric_df.corr()["depression_label"].drop("depression_label")
correlation_with_target = correlation_with_target.sort_values(
    key=abs, ascending=False
)

print(correlation_with_target.round(3))


print("\n" + "=" * 50)
print("COMPARISON BY DEPRESSION LABEL")
print("=" * 50)

features_to_compare = [
    "sleep_hours",
    "daily_social_media_hours",
    "stress_level",
    "anxiety_level",
    "physical_activity",
    "academic_performance"
]

print(
    df.groupby("depression_label")[features_to_compare]
    .mean()
    .round(2)
)


print("\n" + "=" * 50)
print("VALUE RANGE CHECK")
print("=" * 50)

columns_to_check = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level",
    "depression_label"
]

print(df[columns_to_check].agg(["min", "max"]).T)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==================================================
# SECTION 1: LOAD THE DATASET
# ==================================================

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

print("Original dataset shape:", df.shape)


# ==================================================
# SECTION 2: INTRODUCE DATA QUALITY ISSUES
# ==================================================

np.random.seed(50)

# Select 30 rows to duplicate
dup_indices = np.random.choice(df.index, size=30, replace=False)
remaining_indices = list(set(df.index) - set(dup_indices))

# Add 50 missing values to sleep_hours
null_sleep_indices = np.random.choice(
    remaining_indices, size=50, replace=False
)
remaining_indices = list(set(remaining_indices) - set(null_sleep_indices))
df.loc[null_sleep_indices, "sleep_hours"] = np.nan

# Add 40 missing values to academic_performance
null_perf_indices = np.random.choice(
    remaining_indices, size=40, replace=False
)
remaining_indices = list(set(remaining_indices) - set(null_perf_indices))
df.loc[null_perf_indices, "academic_performance"] = np.nan

# Add 5 age outliers
outlier_age_indices = np.random.choice(
    remaining_indices, size=5, replace=False
)
remaining_indices = list(set(remaining_indices) - set(outlier_age_indices))
df.loc[outlier_age_indices, "age"] = 150

# Add 4 social media outliers
outlier_social_indices = np.random.choice(
    remaining_indices, size=4, replace=False
)
df.loc[outlier_social_indices, "daily_social_media_hours"] = 50

# Append 30 duplicate rows
duplicates_to_add = df.loc[dup_indices].copy()
df = pd.concat([df, duplicates_to_add], ignore_index=True)

# Save the dataset containing errors
df.to_csv("Teen_Mental_Health_Dataset_with_Errors.csv", index=False)

print("\nDataset with errors created.")
print("Dataset shape:", df.shape)


# ==================================================
# SECTION 3: INSPECT THE DATASET
# ==================================================

print("\nDataset information:")
df.info()

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())


# ==================================================
# SECTION 4: REMOVE DUPLICATES
# ==================================================

print("\nDuplicates before removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after removal:", df.duplicated().sum())


# ==================================================
# SECTION 5: HANDLE MISSING VALUES
# ==================================================

print("\nMissing values before filling:")
print(df[["sleep_hours", "academic_performance"]].isnull().sum())

# Fill missing sleep hours with the median
df["sleep_hours"] = df["sleep_hours"].fillna(
    df["sleep_hours"].median()
)

# Fill missing academic performance with the median
df["academic_performance"] = df["academic_performance"].fillna(
   df["academic_performance"].median()
)

print("\nMissing values after filling:")
print(df[["sleep_hours", "academic_performance"]].isnull().sum())


# ==================================================
# SECTION 6: DETECT AND REMOVE AGE OUTLIERS
# ==================================================

# Box plot before cleaning
sns.boxplot(y=df["age"])
plt.title("Age Before Outlier Removal")
plt.show()

# Calculate Q1, Q3, and IQR
q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)
iqr = q3 - q1

# Calculate lower and upper limits
lower_limit = q1 - (1.5 * iqr)
upper_limit = q3 + (1.5 * iqr)

print("\nRows before age cleaning:", len(df))

# Remove age outliers
df = df.loc[
    (df["age"] >= lower_limit) &
    (df["age"] <= upper_limit)
].copy()

print("Rows after age cleaning:", len(df))

# Box plot after cleaning
sns.boxplot(y=df["age"])
plt.title("Age After Outlier Removal")
plt.show()


# ==================================================
# SECTION 7: DETECT AND REMOVE SOCIAL MEDIA OUTLIERS
# ==================================================

# Box plot before cleaning
sns.boxplot(y=df["daily_social_media_hours"])
plt.title("Social Media Hours Before Outlier Removal")
plt.show()

# Calculate Q1, Q3, and IQR
q1 = df["daily_social_media_hours"].quantile(0.25)
q3 = df["daily_social_media_hours"].quantile(0.75)
iqr = q3 - q1

# Calculate lower and upper limits
lower_limit = q1 - (1.5 * iqr)
upper_limit = q3 + (1.5 * iqr)

print("\nRows before social media cleaning:", len(df))

# Remove social media outliers
df = df.loc[
    (df["daily_social_media_hours"] >= lower_limit) &
    (df["daily_social_media_hours"] <= upper_limit)
].copy()

print("Rows after social media cleaning:", len(df))

# Box plot after cleaning
sns.boxplot(y=df["daily_social_media_hours"])
plt.title("Social Media Hours After Outlier Removal")
plt.show()


# ==================================================
# SECTION 8: VALIDATE THE CLEANED DATASET
# ==================================================

print("\n========== FINAL DATA QUALITY CHECK ==========")

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows remaining:", df.duplicated().sum())

print("\nAge range:")
print(df["age"].min(), "to", df["age"].max())

print("\nSocial media hours range:")
print(
    df["daily_social_media_hours"].min(),
    "to",
    df["daily_social_media_hours"].max()
)

print("\nFinal dataset shape:", df.shape)

print("\nFinal statistical summary:")
print(df.describe())


# ==================================================
# SECTION 9: SAVE THE CLEANED DATASET
# ==================================================

df.to_csv("Teen_Mental_Health_Dataset_Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File: Teen_Mental_Health_Dataset_Cleaned.csv")
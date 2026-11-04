
# ============================================================
# TEEN MENTAL HEALTH RISK PREDICTION
# Complete Machine Learning Project
#
# Dataset: Teen_Mental_Health_Dataset.csv
# Target: depression_risk (low / medium / high)
#
# Workflow:
# 1. Data Loading
# 2. Data Exploration
# 3. Data Cleaning
# 4. Data Visualization (EDA)
# 5. Train/Test Split
# 6. OneHotEncoder
# 7. SMOTE Augmentation to 10,000 Training Samples
# 8. Training Without/With SMOTE
# 9. Model Evaluation
# 10. Six-Model Comparison
# 11. Cross Validation
# 12. Hyperparameter Tuning
# 13. Feature Importance
# 14. Save Final Model
# 15. Manual Prediction
# ============================================================


# ============================================================
# 0. INSTALL LIBRARIES
# ============================================================

# Run in Google Colab if needed:
# !pip install -q pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Global Settings
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="deep")

os.makedirs("figures", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Make sure the CSV is uploaded to Google Colab
# or available in the current working directory.

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())


# ============================================================
# 3. DATA EXPLORATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["depression_risk"].value_counts())

print("\nTarget Percentages:")
print(
    df["depression_risk"].value_counts(normalize=True) * 100
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

df_clean = df.copy()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Remove rows with missing values
df_clean = df_clean.dropna()

# Reset index
df_clean = df_clean.reset_index(drop=True)

# Confirm valid target labels
valid_classes = ["low", "medium", "high"]

if not df_clean["depression_risk"].isin(valid_classes).all():
    raise ValueError("Unexpected labels in depression_risk")

print("\nOriginal Dataset:", df.shape)
print("Cleaned Dataset:", df_clean.shape)

print(
    "Remaining Missing Values:",
    df_clean.isnull().sum().sum()
)

print(
    "Remaining Duplicates:",
    df_clean.duplicated().sum()
)


# ============================================================
# 5. DATA VISUALIZATION / EDA
# ============================================================

# All visualizations use the cleaned original dataset.
# They do NOT use synthetic SMOTE-generated records.


# ------------------------------------------------------------
# 5.1 Risk Category Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df_clean,
    x="depression_risk",
    order=["low", "medium", "high"],
    hue="depression_risk",
    hue_order=["low", "medium", "high"],
    palette={
        "low": "#2ecc71",
        "medium": "#f1c40f",
        "high": "#e74c3c"
    },
    legend=False
)

plt.title("Distribution of Depression Risk Categories")
plt.xlabel("Risk Category")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    "figures/risk_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.2 Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df_clean,
    x="age",
    bins=7,
    color="steelblue"
)

plt.title("Age Distribution of Adolescents")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "figures/age_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.3 Daily Social Media Usage
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df_clean,
    x="daily_social_media_hours",
    bins=20,
    kde=True,
    color="mediumpurple"
)

plt.title("Daily Social Media Usage Distribution")
plt.xlabel("Hours per Day")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "figures/social_media_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.4 Sleep Hours Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df_clean,
    x="sleep_hours",
    bins=20,
    kde=True,
    color="teal"
)

plt.title("Sleep Hours Distribution")
plt.xlabel("Sleep Hours")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "figures/sleep_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.5 Average Stress and Anxiety by Risk Category
# ------------------------------------------------------------

stress_anxiety = df_clean.groupby(
    "depression_risk"
)[["stress_level", "anxiety_level"]].mean()

stress_anxiety = stress_anxiety.reindex(
    ["low", "medium", "high"]
)

stress_anxiety.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Average Stress and Anxiety by Risk Category")
plt.xlabel("Risk Category")
plt.ylabel("Average Level")
plt.xticks(rotation=0)
plt.legend(["Stress Level", "Anxiety Level"])
plt.tight_layout()

plt.savefig(
    "figures/stress_anxiety_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.6 Stress Level vs Risk Category
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df_clean,
    x="depression_risk",
    y="stress_level",
    order=["low", "medium", "high"],
    hue="depression_risk",
    palette={
        "low": "#2ecc71",
        "medium": "#f1c40f",
        "high": "#e74c3c"
    },
    legend=False
)

plt.title("Stress Level Across Risk Categories")
plt.xlabel("Risk Category")
plt.ylabel("Stress Level")
plt.tight_layout()

plt.savefig(
    "figures/stress_by_risk.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.7 Boxplots - Potential Outliers
# ------------------------------------------------------------

numeric_features = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level"
]

fig, axes = plt.subplots(
    2, 4,
    figsize=(17, 8)
)

for ax, column in zip(axes.flat, numeric_features):

    sns.boxplot(
        y=df_clean[column],
        ax=ax,
        color="skyblue"
    )

    ax.set_title(column)
    ax.set_ylabel("")

plt.suptitle(
    "Numerical Feature Boxplots",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "figures/outlier_boxplots.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.8 Correlation Heatmap
# ------------------------------------------------------------

correlation_matrix = df_clean[
    numeric_features
].corr()

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig(
    "figures/correlation_heatmap.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.9 Sleep Hours by Risk Category
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df_clean,
    x="depression_risk",
    y="sleep_hours",
    order=["low", "medium", "high"],
    color="skyblue"
)

plt.title("Sleep Hours by Risk Category")
plt.xlabel("Risk Category")
plt.ylabel("Sleep Hours")
plt.tight_layout()

plt.savefig(
    "figures/sleep_by_risk.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.10 Social Media Usage by Risk Category
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df_clean,
    x="depression_risk",
    y="daily_social_media_hours",
    order=["low", "medium", "high"],
    color="mediumpurple"
)

plt.title("Social Media Usage by Risk Category")
plt.xlabel("Risk Category")
plt.ylabel("Hours per Day")
plt.tight_layout()

plt.savefig(
    "figures/social_media_by_risk.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5.11 Platform Usage Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df_clean,
    x="platform_usage",
    hue="depression_risk",
    hue_order=["low", "medium", "high"]
)

plt.title("Platform Usage by Risk Category")
plt.xlabel("Social Media Platform")
plt.ylabel("Number of Records")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "figures/platform_usage.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

X = df_clean.drop(
    "depression_risk",
    axis=1
)

y = df_clean["depression_risk"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
print("Target Classes:", y.unique())


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = encoder.fit_transform(
    X_train[categorical_columns]
)

X_test_encoded = encoder.transform(
    X_test[categorical_columns]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)

X_train_encoded = pd.DataFrame(
    X_train_encoded,
    columns=encoded_columns,
    index=X_train.index
)

X_test_encoded = pd.DataFrame(
    X_test_encoded,
    columns=encoded_columns,
    index=X_test.index
)

X_train_ready = pd.concat(
    [X_train[numeric_columns], X_train_encoded],
    axis=1
)

X_test_ready = pd.concat(
    [X_test[numeric_columns], X_test_encoded],
    axis=1
)

print("\nAfter OneHotEncoder:")
print("Training Shape:", X_train_ready.shape)
print("Testing Shape:", X_test_ready.shape)


# ============================================================
# 9. SMOTE DATA AUGMENTATION
# ============================================================

# Increase training dataset to 10,000 samples.
# The test dataset remains unchanged.

smote = SMOTE(
    sampling_strategy={
        "low": 3334,
        "medium": 3333,
        "high": 3333
    },
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_ready,
    y_train
)

print("\nBefore SMOTE:", X_train_ready.shape)
print("After SMOTE:", X_train_resampled.shape)

print("\nDistribution After SMOTE:")
print(y_train_resampled.value_counts())


# ============================================================
# 10. RANDOM FOREST WITHOUT SMOTE
# ============================================================

rf_params = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

rf_without_smote = RandomForestClassifier(
    **rf_params
)

rf_without_smote.fit(
    X_train_ready,
    y_train
)

pred_without = rf_without_smote.predict(
    X_test_ready
)

accuracy_without = accuracy_score(
    y_test,
    pred_without
)

f1_without = f1_score(
    y_test,
    pred_without,
    average="macro"
)

print("\n=== WITHOUT SMOTE ===")
print("Training Samples:", len(X_train_ready))
print("Testing Accuracy:", round(accuracy_without, 4))
print("Testing Macro F1:", round(f1_without, 4))


# ============================================================
# 11. RANDOM FOREST WITH SMOTE
# ============================================================

rf_model = RandomForestClassifier(
    **rf_params
)

rf_model.fit(
    X_train_resampled,
    y_train_resampled
)

train_pred = rf_model.predict(
    X_train_resampled
)

test_pred = rf_model.predict(
    X_test_ready
)

train_accuracy = accuracy_score(
    y_train_resampled,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

train_f1 = f1_score(
    y_train_resampled,
    train_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)

print("\n=== RANDOM FOREST WITH SMOTE ===")

print(
    "Training Accuracy:",
    round(train_accuracy, 4)
)

print(
    "Testing Accuracy:",
    round(test_accuracy, 4)
)

print(
    "Training Macro F1:",
    round(train_f1, 4)
)

print(
    "Testing Macro F1:",
    round(test_f1, 4)
)

print(
    "Accuracy Difference:",
    round(train_accuracy - test_accuracy, 4)
)


# ============================================================
# 12. BEFORE VS AFTER SMOTE COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "Method": [
        "Without SMOTE",
        "With SMOTE"
    ],
    "Training Samples": [
        len(X_train_ready),
        len(X_train_resampled)
    ],
    "Testing Accuracy": [
        accuracy_without,
        test_accuracy
    ],
    "Testing Macro F1": [
        f1_without,
        test_f1
    ]
})

print("\nBefore vs After SMOTE:")
print(comparison_df.round(4))

accuracy_gain = (
    test_accuracy - accuracy_without
) * 100

f1_gain = (
    test_f1 - f1_without
) * 100

print(
    "\nAccuracy Improvement:",
    round(accuracy_gain, 2),
    "percentage points"
)

print(
    "Macro F1 Improvement:",
    round(f1_gain, 2),
    "percentage points"
)


# ============================================================
# 13. VISUALIZE BEFORE VS AFTER SMOTE
# ============================================================

comparison_plot = comparison_df.set_index(
    "Method"
)[["Testing Accuracy", "Testing Macro F1"]] * 100

comparison_plot.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Model Performance: Before vs After SMOTE")
plt.xlabel("Training Method")
plt.ylabel("Score (%)")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend(["Accuracy", "Macro F1"])
plt.tight_layout()

plt.savefig(
    "figures/before_after_smote.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

labels = ["high", "low", "medium"]

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "figures/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. COMPARE SIX MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        **rf_params
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

for name, model in models.items():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            ),
            (
                "num",
                StandardScaler(),
                numeric_columns
            )
        ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro"
        },
        n_jobs=1
    )

    results.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Macro F1": scores["test_f1_macro"].mean(),
        "F1 Std": scores["test_f1_macro"].std()
    })

    print(name, "completed!")


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Macro F1",
    ascending=False
).reset_index(drop=True)

print("\nModel Comparison Results:")
print(results_df.round(4))

best_model_name = results_df.loc[0, "Model"]

print("\nBest Model:", best_model_name)

print(
    "Best CV Macro F1:",
    round(results_df.loc[0, "CV Macro F1"], 4)
)


# ============================================================
# 17. VISUALIZE MODEL COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

sns.barplot(
    data=results_df,
    x="CV Macro F1",
    y="Model",
    color="steelblue"
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Cross Validation Macro F1")
plt.ylabel("Model")
plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    "figures/model_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. FIVE-FOLD CROSS VALIDATION
# ============================================================

cv_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        ("num", "passthrough", numeric_columns)
    ]
)

cv_pipeline = Pipeline([
    ("encoding", cv_preprocessor),

    (
        "smote",
        SMOTE(
            sampling_strategy={
                "low": 3334,
                "medium": 3333,
                "high": 3333
            },
            random_state=RANDOM_STATE
        )
    ),

    (
        "model",
        RandomForestClassifier(
            **rf_params
        )
    )
])

cv_results = cross_validate(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    },
    n_jobs=-1
)

print("\nFive-Fold Cross Validation:")

print(
    "Average CV Accuracy:",
    round(cv_results["test_accuracy"].mean(), 4)
)

print(
    "Average CV Macro F1:",
    round(cv_results["test_f1_macro"].mean(), 4)
)

print(
    "F1 Standard Deviation:",
    round(cv_results["test_f1_macro"].std(), 4)
)


# ============================================================
# 19. HYPERPARAMETER TUNING
# ============================================================

# This section is computationally expensive.
# Set RUN_TUNING = False to skip it when not needed.

RUN_TUNING = True

if RUN_TUNING:

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [5, 8, 10, 12, None],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", "log2", 0.8]
    }

    random_search = RandomizedSearchCV(
        estimator=cv_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    random_search.fit(
        X_train,
        y_train
    )

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest CV Macro F1:")
    print(round(random_search.best_score_, 4))


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train_ready.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
    color="teal"
)

plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    "figures/feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. TRAIN AND SAVE FINAL MODEL
# ============================================================

# We keep the baseline Random Forest settings.
# The pipeline includes OneHotEncoder and SMOTE.
# SMOTE is active during training, not prediction.

final_model = cv_pipeline.set_params(
    model__n_estimators=300,
    model__max_depth=8,
    model__min_samples_split=10,
    model__min_samples_leaf=3,
    model__max_features="sqrt"
)

final_model.fit(
    X_train,
    y_train
)

joblib.dump(
    final_model,
    "depression_risk_model.pkl"
)

print("\nFinal Model Saved Successfully!")


# ============================================================
# 22. MANUAL PREDICTION
# ============================================================

loaded_model = joblib.load(
    "depression_risk_model.pkl"
)

new_data = pd.DataFrame([{
    "age": 16,
    "gender": "male",
    "daily_social_media_hours": 5.5,
    "platform_usage": "Instagram",
    "sleep_hours": 6.0,
    "screen_time_before_sleep": 2.0,
    "academic_performance": 3.0,
    "physical_activity": 1.0,
    "social_interaction_level": "medium",
    "stress_level": 7,
    "anxiety_level": 8
}])

prediction = loaded_model.predict(
    new_data
)[0]

probabilities = loaded_model.predict_proba(
    new_data
)[0]

print("\nPredicted Risk Category:", prediction)

probability_df = pd.DataFrame({
    "Risk Category": loaded_model.classes_,
    "Model Score (%)": probabilities * 100
})

print("\nModel Scores:")
print(probability_df.round(2))


# ============================================================
# 23. VISUALIZE MANUAL PREDICTION SCORES
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=probability_df,
    x="Risk Category",
    y="Model Score (%)",
    hue="Risk Category",
    palette={
        "low": "#2ecc71",
        "medium": "#f1c40f",
        "high": "#e74c3c"
    },
    legend=False
)

plt.title("Predicted Risk Category Scores")
plt.ylabel("Model Score (%)")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "figures/manual_prediction.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("Original Records:", len(df_clean))
print("Training Records Before SMOTE:", len(X_train))
print("Training Records After SMOTE:", len(X_train_resampled))
print("Testing Records:", len(X_test))

print("\nSelected Algorithm: Random Forest")

print(
    "Accuracy Without SMOTE:",
    round(accuracy_without * 100, 2), "%"
)

print(
    "Accuracy With SMOTE:",
    round(test_accuracy * 100, 2), "%"
)

print(
    "Macro F1 With SMOTE:",
    round(test_f1 * 100, 2), "%"
)

print(
    "Accuracy Improvement:",
    round(accuracy_gain, 2),
    "percentage points"
)

print("\nSaved Model: depression_risk_model.pkl")
print("Saved Figures: figures/")

print(
    "\nDISCLAIMER: This is an educational, dataset-based "
    "classification project, not a medical diagnostic tool."
)

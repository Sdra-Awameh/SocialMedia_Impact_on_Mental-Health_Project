# Social Media Impact on Teen Mental Health

<p align="center">
  **A Machine Learning Approach to Predicting Depression Risk Among Teenagers**
</p>

<p align="center">
  ![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn)
  ![Project](https://img.shields.io/badge/Project-Educational-purple)
</p>

## Overview

Social media has become an essential part of teenagers' everyday lives. Their social media habits, sleep patterns, academic performance, and lifestyle choices may be associated with different mental health outcomes.

This project investigates the relationship between social media usage and teenagers' mental health using **Data Analysis and Machine Learning** techniques.

The main objective is to develop a classification model that predicts depression risk categories based on behavioral, lifestyle, and psychological indicators.

The prediction task includes three categories:

- **Low Risk**
- **Medium Risk**
- **High Risk**

> **Disclaimer:** This project is intended for educational and analytical purposes only. It does not provide medical diagnoses or replace professional mental health assessment.

## Project Objectives

- Investigate patterns between social media usage and mental health indicators.
- Explore the relationship between lifestyle habits and depression risk.
- Perform data cleaning, preprocessing, and Exploratory Data Analysis (EDA).
- Apply techniques to address class imbalance in the training data.
- Build and compare multiple Machine Learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Identify influential features in the selected model.
- Analyze prediction errors and identify areas for improvement.

## Dataset

The dataset contains **2,500 records and 12 columns**, including behavioral, lifestyle, and psychological information.

### Features Description

| Feature | Description |
|---|---|
| `age` | Teenager's age |
| `gender` | Gender |
| `daily_social_media_hours` | Average daily social media usage |
| `platform_usage` | Social media platform used |
| `sleep_hours` | Average daily sleep duration |
| `screen_time_before_sleep` | Screen time before sleeping |
| `academic_performance` | Academic performance level |
| `physical_activity` | Physical activity level |
| `social_interaction_level` | Level of social interaction |
| `stress_level` | Reported stress level |
| `anxiety_level` | Reported anxiety level |
| `depression_risk` | Target variable representing depression risk category |

### Target Distribution

The target variable, `depression_risk`, contains three classes:

| Risk Category | Records | Distribution |
|---|---:|---:|
| Low | 1,266 | 50.64% |
| Medium | 650 | 26.00% |
| High | 584 | 23.36% |
| **Total** | **2,500** | **100%** |

The original dataset has an imbalanced class distribution, which is addressed during model training using SMOTE.

## Project Methodology

The project follows a structured Machine Learning workflow.

### 1. Data Loading and Exploration

The dataset is loaded using Pandas and inspected to understand its structure and characteristics.

The exploration includes:

- Dataset dimensions and column information.
- Descriptive statistics.
- Missing value detection.
- Duplicate record detection.
- Target class distribution.

### 2. Data Cleaning

The dataset is cleaned by:

- Removing duplicate records.
- Removing rows containing missing values.
- Resetting the dataset index.
- Validating the target labels.

### 3. Exploratory Data Analysis (EDA)

EDA is performed using Matplotlib and Seaborn to explore patterns and relationships in the dataset.

The visualizations include:

- Risk category distribution.
- Age distribution.
- Daily social media usage distribution.
- Sleep hours distribution.
- Average stress and anxiety by risk category.
- Stress levels across risk categories.
- Numerical feature boxplots.
- Correlation heatmap.
- Sleep hours by risk category.
- Social media usage by risk category.
- Platform usage by risk category.

The analysis helps identify differences in behavioral and lifestyle indicators across depression risk categories.

These observations represent associations within the dataset and should not be interpreted as causal relationships.

### 4. Features and Target Selection

The dataset is separated into:

- **Features (X):** Behavioral, lifestyle, and psychological input variables.
- **Target (y):** `depression_risk`.

### 5. Train-Test Split

The cleaned dataset is divided into training and testing sets using a **70/30 split**.

| Dataset | Percentage |
|---|---:|
| Training Set | 70% |
| Testing Set | 30% |

A **Stratified Split** is used to preserve the target class distribution in both sets.

The testing data remains separate and is not augmented with SMOTE.

### 6. Data Preprocessing

Categorical variables are transformed using **One-Hot Encoding**.

The preprocessing includes:

- Identifying categorical and numerical columns.
- Applying `OneHotEncoder` to categorical features.
- Keeping numerical features in the prepared dataset.
- Applying preprocessing through pipelines during model comparison and cross-validation.

### 7. Handling Class Imbalance with SMOTE

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance in the training data.

SMOTE generates synthetic training samples to create a balanced training dataset containing **10,000 records**:

| Class | Training Records After SMOTE |
|---|---:|
| Low | 3,334 |
| Medium | 3,333 |
| High | 3,333 |
| **Total** | **10,000** |

SMOTE is applied only to training data. The original testing set remains unchanged to provide an independent evaluation.

### 8. Random Forest: Before and After SMOTE

Random Forest is evaluated in two scenarios:

- Training using the original imbalanced training data.
- Training using the SMOTE-resampled training data.

The results are compared using:

- Testing Accuracy.
- Testing Macro F1-score.

This comparison helps examine the effect of SMOTE on model performance.

## Machine Learning Models

Six classification algorithms are compared using cross-validation:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. HistGradientBoosting

The models are evaluated using a **5-Fold Stratified Cross-Validation** strategy.

The comparison focuses on:

- Cross-validation Accuracy.
- Cross-validation Macro F1-score.
- Standard deviation of Macro F1-score.

The models are ranked based on their cross-validation Macro F1-score.

## Model Evaluation

The project uses several evaluation metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Macro F1-score:** Average F1-score across all classes, treating each class equally.
- **Weighted F1-score:** Average F1-score weighted by the number of samples in each class.
- **Precision:** Proportion of predicted positive class labels that are correct.
- **Recall:** Proportion of actual class records correctly identified.
- **Confusion Matrix:** Shows correct and incorrect classifications across the three classes.
- **Classification Report:** Provides precision, recall, and F1-score for each class.

The Random Forest model is evaluated using the unchanged testing dataset.

## Cross-Validation

A **5-Fold Stratified Cross-Validation** process is used to evaluate the Random Forest pipeline.

The process includes:

- Categorical feature encoding.
- SMOTE applied within each training fold.
- Random Forest model training.
- Accuracy and Macro F1-score evaluation.

This helps assess model performance across different training and validation splits.

## Hyperparameter Tuning

The project applies **RandomizedSearchCV** to explore different Random Forest hyperparameter combinations.

The tuning process uses:

- 5-Fold Stratified Cross-Validation.
- Macro F1-score as the optimization metric.
- 20 randomly selected parameter combinations.

The search explores parameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

The best parameters and cross-validation score are displayed after the search.

The final saved model in the current script uses the defined baseline Random Forest settings.

## Feature Importance Analysis

Feature importance is extracted from the trained Random Forest model.

This analysis helps identify which input features contribute most to the model's predictions.

The feature importance results are visualized using a bar chart.

**Important:** Feature importance represents predictive contribution within the model. It does not establish causation or prove that a particular feature causes depression.

## Final Model and Prediction

The final Random Forest pipeline includes:

- One-Hot Encoding.
- SMOTE during training.
- Random Forest classification.

The trained pipeline is saved using Joblib as:

`depression_risk_model.pkl`

The project also includes a manual prediction example using a sample teenager profile.

The model returns:

- Predicted depression risk category.
- Model scores for Low, Medium, and High risk categories.

These scores represent model outputs, not clinical probabilities or diagnoses.

## Output Files

When `main.py` is executed successfully, the project generates:

| File / Folder | Description |
|---|---|
| `figures/` | Generated EDA and model evaluation visualizations |
| `depression_risk_model.pkl` | Saved final Machine Learning pipeline |

The figures include risk distribution, feature distributions, correlation heatmap, SMOTE comparison, confusion matrix, model comparison, feature importance, and manual prediction scores.

## Technologies & Libraries

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Imbalanced-learn | SMOTE and Machine Learning pipelines |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Exploratory analysis and experimentation |

## Project Structure

```text
SocialMedia_Impact_on_Teen_Mental_Health/
│
├── data/
│   ├── raw/
│   │   └── Teen_Mental_Health_Dataset.csv
│   │
│   └── processed/
│       ├── Teen_Mental_Health_Dataset_Cleaned.csv
│       └── Teen_Mental_Health_Dataset_with_Errors.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── main.py
├── README.md
│
├── figures/                     # Generated when main.py runs
└── depression_risk_model.pkl    # Generated when main.py runs
```

## Installation and Usage

### Prerequisites

- Python 3.x
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Sdra-Awameh/SocialMedia_Impact_on_Mental-Health_Project.git
cd SocialMedia_Impact_on_Mental-Health_Project
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 3. Check the Dataset Path

The current `main.py` script reads the dataset from the project root:

```python
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
```

If the dataset is stored inside `data/raw/`, update the line to:

```python
df = pd.read_csv("data/raw/Teen_Mental_Health_Dataset.csv")
```

### 4. Run the Main Script

```bash
python main.py
```

### 5. Open the EDA Notebook

Open `notebooks/EDA.ipynb` using Jupyter Notebook or VS Code to explore the data analysis.

## Limitations

Despite the structured workflow, the project has several limitations:

- The dataset may not fully represent the diversity of teenagers and real-world situations.
- SMOTE creates synthetic training samples and does not replace collecting additional real-world data.
- The Medium-risk category may be challenging to distinguish from the Low and High categories.
- Model performance depends on the quality and representativeness of the dataset.
- Feature importance does not establish causal relationships.
- The model is not designed or validated for clinical diagnosis.

## Future Improvements

Potential directions for future development include:

- Collecting a larger and more diverse real-world dataset.
- Improving classification performance across all risk categories.
- Exploring additional Machine Learning algorithms.
- Further optimizing model hyperparameters.
- Evaluating the model using an independent external dataset.
- Applying model interpretability techniques for deeper analysis.
- Improving the prediction pipeline and deployment options.

## Conclusion

This project explores the relationship between social media usage, lifestyle habits, and depression risk among teenagers using Data Analysis and Machine Learning.

It includes data cleaning, Exploratory Data Analysis, preprocessing, class imbalance handling with SMOTE, model comparison, cross-validation, hyperparameter tuning, feature importance analysis, and manual prediction.

The project demonstrates how Machine Learning techniques can be used to identify patterns in mental-health-related data. However, further improvements and independent validation are needed before such a model could be considered for any real-world application.

---

*This project was developed for educational and research purposes. It is not a clinical diagnostic system and must not be used as a substitute for professional mental health assessment.*

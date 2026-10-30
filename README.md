# Social Media Impact on Teen Mental Health

<p align="center">
  <strong>A Machine Learning Approach to Predicting Depression Risk Among Teenagers</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project Status">
  <img src="https://img.shields.io/badge/Project-Educational-purple" alt="Educational Project">
</p>

## Overview

Social media has become an essential part of teenagers' everyday lives. However, excessive social media usage, sleep patterns, academic pressure, and lifestyle habits may be associated with mental health challenges.

This project investigates the relationship between social media usage and teenagers' mental health using **Data Analysis and Machine Learning** techniques.

The main objective is to develop a classification model that predicts depression risk levels based on behavioral, lifestyle, and psychological indicators.

The prediction task includes three categories:

- **Low Risk**
- **Medium Risk**
- **High Risk**

> This project is intended for educational and analytical purposes. It does not provide medical diagnoses or replace professional mental health assessment.

## Project Objectives

- Investigate patterns between social media usage and mental health indicators.
- Explore the relationship between lifestyle habits and depression risk.
- Perform data cleaning, preprocessing, and exploratory data analysis.
- Build and compare multiple Machine Learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Identify the most influential features in the selected model.
- Analyze prediction errors and identify areas for improvement.

## Dataset

The dataset contains **2,500 records and 12 features**, including behavioral, lifestyle, and psychological information.

### Features Description

| Feature | Description |
|---|---|
| `age` | Teenager's age |
| `gender` | Gender |
| `daily_social_media_hours` | Average daily social media usage |
| `platform_usage` | Social media platforms used |
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

| Risk Category | Number of Records | Distribution |
|---|---:|---:|
| Low | 1,266 | 50.64% |
| Medium | 650 | 26.00% |
| High | 584 | 23.36% |
| **Total** | **2,500** | **100%** |

## Methodology

The project followed a structured Machine Learning workflow.

### 1. Data Cleaning

The dataset was inspected to ensure data quality.

- Checked for missing values.
- Checked for duplicate records.
- Reviewed dataset structure and feature types.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to identify patterns and differences between depression risk categories.

Some key observations included:

- The High-risk group had higher average daily social media usage.
- The High-risk group showed lower average sleep duration.
- Higher average stress and anxiety levels were observed in the High-risk group.
- Academic performance and physical activity also showed differences across risk categories.

These observations represent associations within the dataset and should not be interpreted as causal relationships.

### 3. Data Preprocessing

Categorical features were handled according to their data types:

- **One-Hot Encoding:** Applied to nominal features such as `gender` and `platform_usage`.
- **Ordinal Encoding:** Applied to `social_interaction_level`, preserving its order: Low, Medium, High.
- **Feature Scaling:** Applied where required by the selected algorithms.

### 4. Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split.

| Dataset | Records | Percentage |
|---|---:|---:|
| Training Set | 2,000 | 80% |
| Testing Set | 500 | 20% |

A **Stratified Split** was used to preserve the target class distribution in both sets.

## Machine Learning Models

Four classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. K-Nearest Neighbors (KNN)

## Model Evaluation

The models were evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Macro F1-score:** Average F1-score across classes, treating each class equally.
- **Weighted F1-score:** Average F1-score weighted by the number of samples in each class.
- **Confusion Matrix:** Used to examine correct and incorrect classifications.
- **Classification Report:** Used to evaluate precision, recall, and F1-score for each class.

### Test Set Performance

| Model | Accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| Logistic Regression | 71.8% | 0.67 | 0.71 |
| Random Forest | **75.0%** | **0.72** | **0.75** |
| Decision Tree | 73.2% | 0.69 | 0.73 |
| KNN | 70.4% | 0.64 | 0.69 |

### Cross-Validation Performance

To evaluate model stability, **5-Fold Stratified Cross-Validation** was performed using the training data.

| Model | Accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| Logistic Regression | 74.00% | 0.691 | 0.736 |
| Random Forest | **76.60%** | **0.726** | **0.761** |
| Decision Tree | 74.65% | 0.703 | 0.747 |
| KNN | 72.75% | 0.672 | 0.719 |

## Final Model Selection: Random Forest

Based on the test set and cross-validation results, **Random Forest** achieved the strongest overall performance among the four evaluated models.

### Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 75.00% |
| Macro F1-score | 0.72 |
| Weighted F1-score | 0.75 |

The similarity between cross-validation and test results suggests relatively consistent model performance.

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| High | 0.70 | 0.88 | 0.78 |
| Low | 0.92 | 0.81 | 0.86 |
| Medium | 0.52 | 0.52 | 0.52 |

The model performed better when identifying Low and High risk categories, while the Medium category remained more challenging.

## Feature Importance Analysis

Feature importance was extracted from the trained Random Forest model to identify which features contributed most to its predictions.

### Feature Importance Ranking

| Feature | Importance |
|---|---:|
| Stress Level | 22.48% |
| Academic Performance | 16.76% |
| Daily Social Media Hours | 14.24% |
| Anxiety Level | 13.34% |
| Sleep Hours | 10.08% |
| Physical Activity | 5.78% |
| Screen Time Before Sleep | 5.69% |
| Age | 3.94% |

The results highlight stress level, academic performance, daily social media usage, anxiety level, and sleep duration as the most influential features in the model.

**Important:** Feature importance reflects predictive contribution within the model. It does not establish causation or prove that a particular feature causes depression.

## Error Analysis

The selected Random Forest model produced **125 incorrect predictions out of 500 test records**.

The main classification errors were:

| Actual Class | Predicted Class | Number of Errors |
|---|---|---:|
| High | Medium | 14 |
| Low | Medium | 48 |
| Medium | High | 44 |
| Medium | Low | 19 |

The Medium-risk category was the most challenging for the model to classify correctly.

This suggests that some Medium-risk records share patterns with both Low-risk and High-risk records, making the boundaries between these categories harder to distinguish.

## Limitations

Despite achieving the best overall performance among the evaluated models, this project has several limitations:

- The final model achieved 75% test accuracy, leaving room for improvement.
- The Medium-risk category had a relatively low F1-score of 0.52.
- The dataset may not fully represent the diversity of teenagers and real-world situations.
- Feature importance does not establish causal relationships.
- The model is not designed or validated for clinical diagnosis.

## Future Improvements

Potential directions for future development include:

- Collecting a larger and more diverse dataset.
- Improving the classification performance of the Medium-risk category.
- Applying hyperparameter tuning to optimize model performance.
- Exploring additional Machine Learning algorithms.
- Investigating additional relevant behavioral and lifestyle features.
- Evaluating the model using an independent external dataset.
- Applying model interpretability techniques for deeper analysis.

## Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Jupyter Notebook | Development and experimentation |

## Project Workflow

1. Dataset inspection and cleaning
2. Exploratory Data Analysis
3. Feature identification and preprocessing
4. Train-test splitting
5. Machine Learning model development
6. Model evaluation and comparison
7. Cross-validation
8. Final model selection
9. Feature importance analysis
10. Error analysis and documentation

## Conclusion

This project explored the relationship between social media usage, lifestyle habits, and depression risk among teenagers using Machine Learning.

Four classification algorithms were implemented and compared. Random Forest achieved the best overall results, with **75% accuracy on the test set**.

The analysis identified stress level, academic performance, social media usage, anxiety level, and sleep duration as the most influential predictive features.

Although the results demonstrate the potential of Machine Learning for analyzing patterns in mental health-related data, further work is needed to improve classification performance, particularly for the Medium-risk category.

---

## Author

**Sdra Awameh**

Software Engineering Student

---

*Developed for educational and research purposes. This project is not a clinical diagnostic system and must not be used as a substitute for professional mental health assessment.*

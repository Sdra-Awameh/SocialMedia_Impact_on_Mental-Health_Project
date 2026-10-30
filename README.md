# 🧠 Social Media Impact on Teen Mental Health

## 💼 Business Problem

Excessive social media use may be associated with sleep disturbances, increased stress and anxiety, and changes in teenagers' mental well-being. Understanding these relationships can help identify patterns and factors associated with depression risk.

This project analyzes teenagers' social media habits, sleep patterns, and other lifestyle factors to explore their potential relationships with mental health outcomes.

## 🎯 Objectives

- Analyze teenagers' social media usage patterns.
- Examine relationships between social media usage, sleep, stress, and anxiety.
- Identify factors associated with depression risk.
- Discover meaningful patterns through Exploratory Data Analysis (EDA).
- Create visualizations to communicate key findings.
- Prepare and evaluate data for a potential depression prediction model.

## 🔮 Prediction Target

**Target Variable:** Depression-risk

The goal is to investigate whether depression status can be predicted using available features such as social media usage, sleep patterns, lifestyle habits, and other relevant factors in the dataset.

The feasibility of prediction depends on the available target variable, feature quality, and model evaluation results.

## 📊 Dataset

| Attribute | Description |
|---|---|
| **Dataset Name** | Teen Mental Health Dataset — Teen_Mental_Health_Dataset |
| **Source** | [Kaggle — Teen_Mental_Health_Dataset](https://www.kaggle.com/datasets/sunil123kumar/social-media-impact-on-mental-health) |
| **Size** | 1,200 rows × 13 columns (original dataset) |
| **Age Range** | 13–19 years |
| **Missing Values** | Reported as none in the original dataset |
| **Data Type** | Structured tabular data |

## 🧹 Data Cleaning and Preprocessing

The project includes the following data quality checks and preprocessing steps:

- Check for duplicate records.
- Inspect missing values.
- Handle missing values when present.
- Detect potential outliers in selected numerical columns.
- Validate the cleaned dataset.
- Save the processed dataset for further analysis.

## 📁 Project Structure

```text
SocialMedia_Impact_on_Teen_Mental_Health/
├── data/
│   ├── raw/
│   │   └── Teen_Mental_Health_Dataset.csv
│   └── processed/
│       ├── Teen_Mental_Health_Dataset_Cleaned.csv
│       └── Teen_Mental_Health_Dataset_with_Errors.csv
├── notebooks/
│   └── EDA.ipynb
├── main.py
└── README.md
```

## ▶️ Running the Analysis

Run the analysis script from the project root:

```bash
python main.py
```

Open `notebooks/EDA.ipynb` to explore the dataset interactively.

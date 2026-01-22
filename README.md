# Insurance Claim Prediction & Risk Analysis (MediBuddy Capstone)

End-to-end machine learning project to predict healthcare insurance claim amounts and identify key risk drivers for data-driven premium pricing.

---

## Overview

This project analyzes healthcare insurance data and builds regression models to predict claim amounts based on demographic and medical factors.  
The goal is to support automated premium estimation and insurance risk assessment.

Key tasks:
- Data merging and cleaning  
- Exploratory data analysis  
- Feature engineering and preprocessing pipelines  
- Model training and comparison  
- Business insight generation  

---

## Dataset

Two datasets merged using **Policy Number**.

**Personal Details**
- Gender  
- Region  
- Number of children  
- Smoking status  

**Medical & Cost**
- Age  
- BMI  
- Insurance charges (target)  

- Total records: **1,338**  
- Target variable: **Charges (INR)**  

---

## Exploratory Data Analysis

Main observations:
- Insurance charges are highly right-skewed with extreme outliers  
- Smokers incur dramatically higher claim amounts  
- Age and BMI strongly increase insurance cost  
- Gender and region have negligible impact  

Visual analysis included:
- Distributions of age, BMI, and charges  
- Boxplots by smoker status, gender, region, and dependents  
- Interaction plots (Age × Smoker, BMI × Smoker)  
- Correlation heatmaps and interactive Plotly charts  

---

## Preprocessing & Feature Engineering

- Merged datasets using policy number  
- Removed identifiers and leakage-prone variables  
- One-hot encoded categorical features  
- Numerical scaling where required  
- Built unified preprocessing + modeling pipelines using:
  - `ColumnTransformer`  
  - `Pipeline`  

Train–test split: **80 / 20**

---

## Models & Performance

| Model              | R²     | RMSE (INR) |
|--------------------|--------|------------|
| Linear Regression  | 0.784  | 5,796      |
| Random Forest      | 0.864  | 4,594      |
| Gradient Boosting  | **0.879** | **4,335** |

**Final model selected:** Gradient Boosting Regressor

---

## Feature Importance (Gradient Boosting)

Primary risk drivers:
- Smoking status → **~68%**  
- BMI → **~19%**  
- Age → **~12%**  
- Children → ~1%  

Negligible impact:
- Gender  
- Region  

---

## Business Insights

- Smoking status is the dominant predictor of insurance cost  
- BMI and age are strong medical risk indicators  
- Gender and geographic region add little pricing value  
- Number of dependents has minimal effect  

---

## Recommendations

- Introduce significant smoker surcharges  
- Apply BMI-based premium adjustments  
- Increase premiums progressively with age  
- Offer discounts to non-smokers with healthy BMI  
- Avoid pricing based on gender or region  
- Deploy Gradient Boosting for automated premium estimation  

---

## Tools & Technologies

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- Scikit-learn  
- Pipelines, ColumnTransformer, GridSearchCV  

---

## Notebook

Interactive version:  
(link will be added here)


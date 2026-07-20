# Healthcare Insurance Risk Modelling

An end-to-end Machine Learning project that predicts healthcare insurance charges using demographic and lifestyle information. The project applies Exploratory Data Analysis (EDA), data preprocessing, feature engineering, multiple regression models, model evaluation, and business interpretation to support data-driven insurance premium estimation.

---

##  Project Overview

Healthcare insurance providers need accurate premium estimation to reduce financial risk while offering competitive pricing. This project develops and compares multiple machine learning regression models to predict insurance charges and identify the key factors influencing insurance costs.

The final model provides accurate premium predictions and actionable business recommendations for insurance companies.

---

##  Objectives

- Predict healthcare insurance charges accurately.
- Analyze factors affecting insurance costs.
- Compare multiple machine learning regression models.
- Select the best-performing model.
- Generate business insights and recommendations for premium pricing.

---

##  Dataset Information

The dataset contains customer demographic and health-related information.

### Features

| Feature | Description |
|----------|-------------|
| Age | Age of the insured person |
| Sex | Male/Female |
| BMI | Body Mass Index |
| Children | Number of dependents |
| Smoker | Smoking status |
| Region | Residential region |
| Charges | Insurance charges (Target Variable) |

---

#  Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

#  Project Workflow

```
Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Preprocessing
     │
     ▼
Train-Test Split
     │
     ▼
Pipeline + ColumnTransformer
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Feature Importance
     │
     ▼
Business Insights
```

---

#  Exploratory Data Analysis

EDA was performed to understand relationships between variables and identify the major drivers of insurance cost.

Analysis included:

- Distribution of insurance charges
- Age vs Charges
- BMI vs Charges
- Smokers vs Non-smokers
- Region-wise comparison
- Gender analysis
- Children vs Charges
- Correlation analysis

---

#  Data Preprocessing

The following preprocessing steps were applied:

- Missing value verification
- One-Hot Encoding
- ColumnTransformer
- Train-Test Split (80:20)
- Scikit-learn Pipeline implementation

---

#  Machine Learning Models

The following models were trained and evaluated.

### 1. Linear Regression

Baseline regression model used for comparison.

---

### 2. Random Forest Regressor

Ensemble learning model capable of capturing non-linear relationships.

---

### 3. Gradient Boosting Regressor

Boosting-based ensemble model providing the best predictive performance.

---

### 4. Hyperparameter Tuned Gradient Boosting

GridSearchCV was used for hyperparameter optimization.

---

#  Model Performance

| Model | R² Score | RMSE |
|--------|---------:|------:|
| Linear Regression | **0.784** | **5796** |
| Random Forest | **0.864** | **4594** |
| Gradient Boosting | **0.879** | **4335** |
| Tuned Gradient Boosting | **0.878** | **4344** |

---

#  Final Model

**Gradient Boosting Regressor**

### Performance

- **R² Score:** 0.879
- **RMSE:** 4335

Although GridSearchCV was performed, the tuned model did not improve performance on the test dataset. Therefore, the original Gradient Boosting model was selected as the final production model.

---

#  Feature Importance

| Feature | Importance |
|---------|-----------:|
| Smoker | 67.7% |
| BMI | 19.0% |
| Age | 11.9% |
| Children | <1% |
| Gender | Negligible |
| Region | Negligible |

---

#  Model Diagnostics

Model validation included:

- Actual vs Predicted Plot
- Residual Distribution
- Residual vs Predicted Plot

These diagnostics indicate that the model produces accurate predictions with no significant systematic error patterns.

---

#  Business Insights

- Smoking status is the strongest determinant of insurance cost.
- BMI is the second most influential pricing factor.
- Insurance charges increase steadily with age.
- Number of dependents has minimal influence.
- Gender has negligible impact on insurance cost.
- Geographic region contributes very little to prediction accuracy.

---

#  Business Recommendations

- Use smoking status as the primary premium pricing factor.
- Incorporate BMI and age into pricing models.
- Offer premium discounts to healthy non-smokers.
- Avoid pricing decisions based primarily on gender or region.
- Deploy Gradient Boosting for automated premium estimation.

---

#  Repository Structure

```
Healthcare-Insurance-Risk-Modelling/
│
├── Dataset/
│
├── Notebook/
│   └── Healthcare_Insurance_Risk_Modelling.ipynb
│
├── Images/
│
├── Report/
│   └── Healthcare_Insurance_Risk_Modelling_Report.pdf
│
├── README.md
│
└── requirements.txt
```

---

#  Future Improvements

- Deploy the model using Streamlit
- Develop a premium prediction web application
- Integrate SHAP for explainable AI
- Train using larger healthcare datasets
- Implement automated model monitoring

---

#  Report

A detailed project report is included in the repository.

```
Report/
└── Healthcare_Insurance_Risk_Modelling_Report.pdf
```

---

#  Author

**Nishant Pratap Singh**

**GitHub:** https://github.com/npstanwar

**LinkedIn:** https://www.linkedin.com/in/npstanwar/

---

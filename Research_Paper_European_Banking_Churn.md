# Customer Segmentation & Churn Pattern Analytics in European Banking
### Research Paper | Unified Mentor Data Analytics Project

---

**Author:** Sumit Sharma  
**Organization:** Unified Mentor  
**Dataset:** European Bank Customer Dataset (10,000 records)  
**Tools Used:** Python, Pandas, Scikit-learn, Matplotlib, Streamlit  
**Date:** November 2026

---

## Abstract

Customer churn is one of the most significant threats to revenue stability in retail banking. This paper presents a comprehensive, segmentation-driven analysis of customer churn across three European geographies — France, Germany, and Spain — using a dataset of 10,000 bank customers. Through systematic data validation, feature engineering, exploratory data analysis (EDA), KPI computation, and machine learning modelling, we identify the key drivers and high-risk customer segments that contribute disproportionately to churn. Our findings reveal that Germany carries the highest geographic churn risk, the 46–60 age cohort exhibits the sharpest churn tendency, high-balance customers churn at a significantly elevated rate, and member inactivity is the strongest single-variable predictor of churn. A Random Forest classifier achieves the best predictive performance, with actionable retention strategies proposed for each high-risk segment.

---

## 1. Introduction

### 1.1 Background

Customer retention is fundamentally more cost-effective than customer acquisition. Industry studies consistently show that acquiring a new bank customer costs five to seven times more than retaining an existing one. Despite this, many banks manage churn reactively — responding only after a customer has already left — rather than proactively identifying at-risk customers and intervening in advance.

European retail banks face a particularly complex churn landscape. Regulatory diversity, geographic competition, digital banking disruption, and varying customer expectations across France, Germany, and Spain create a multi-dimensional churn problem that generic, one-size-fits-all retention strategies cannot adequately address.

### 1.2 Problem Statement

This project addresses the following core questions:

1. What is the overall rate of customer churn across the European bank portfolio?
2. Which customer segments — defined by geography, age, credit profile, tenure, and balance — exhibit the highest churn rates?
3. How do churned customers differ in financial and behavioural profile from retained customers?
4. Which segments represent the highest revenue risk due to churn?
5. Can machine learning accurately predict which customers are likely to churn?

### 1.3 Objectives

**Primary Objectives**
- Measure the overall and segment-wise churn rate
- Identify churn distribution across all defined segmentation dimensions
- Compare churn behaviour across European regions

**Secondary Objectives**
- Understand churn concentration among high-value customers
- Evaluate the relationship between engagement, tenure, and churn
- Build predictive models to support proactive retention strategy

---

## 2. Dataset Description

The dataset contains **10,000 customer records** from a European retail bank with 13 original analytical columns:

| Column | Type | Description |
|--------|------|-------------|
| `CustomerId` | Integer | Unique identifier (excluded from predictive modelling) |
| `Surname` | String | Customer surname (excluded — non-analytical) |
| `CreditScore` | Integer | Customer credit score (350–850) |
| `Geography` | Categorical | Country: France, Germany, Spain |
| `Gender` | Categorical | Male / Female |
| `Age` | Integer | Customer age in years |
| `Tenure` | Integer | Years as an account holder (0–10) |
| `Balance` | Float | Current account balance |
| `NumOfProducts` | Integer | Number of bank products held (1–4) |
| `HasCrCard` | Binary | Owns a credit card (0/1) |
| `IsActiveMember` | Binary | Active member indicator (0/1) |
| `EstimatedSalary` | Float | Estimated annual salary |
| `Exited` | Binary | Churn indicator — **target variable** (0 = retained, 1 = churned) |

**Data Quality Assessment**
- No missing values detected across all columns
- Zero duplicate `CustomerId` records confirmed
- All binary variables (`HasCrCard`, `IsActiveMember`, `Exited`) verified as valid 0/1 integers
- Class distribution: 7,963 retained (79.63%), 2,037 churned (20.37%)

---

## 3. Methodology

### 3.1 Data Ingestion & Validation
The dataset was loaded into a Pandas DataFrame from `European_Bank.csv`. Null checks, duplicate detection, and category validations were executed to ensure dataset integrity.

### 3.2 Feature Engineering — Segmentation Dimensions

Four derived categorical segmentation columns were engineered:

| New Column | Source | Segments & Cut-offs |
|------------|--------|---------------------|
| `AgeGroup` | `Age` | `<30`, `30–45`, `46–60`, `60+` |
| `CreditScoreBand` | `CreditScore` | Low (<580), Medium (580–669), High (≥670) |
| `TenureGroup` | `Tenure` | New (0–2 yrs), Mid-term (3–6 yrs), Long-term (7+ yrs) |
| `BalanceSegment` | `Balance` | Zero-balance (=0), Low-balance (>0 ≤ median), High-balance (>median) |

The positive-balance median was calculated exclusively among accounts with balances greater than zero (~€119,839.69), ensuring zero-balance accounts did not skew the threshold.

### 3.3 Key Performance Indicator (KPI) Definitions

| KPI | Formula / Operational Definition |
|-----|----------------------------------|
| **Overall Churn Rate** | `(Total Churned / Total Customers) × 100` |
| **Segment Churn Rate** | Churn rate calculated within each distinct subgroup |
| **High-Value Churn Ratio** | Churn percentage among customers in the `High-balance` segment |
| **Geographic Risk Index** | `Country Churn Rate / Overall Churn Rate` (>1 indicates above-average risk) |
| **Engagement Drop Indicator** | `Inactive Member Churn Rate − Active Member Churn Rate` (percentage points) |

### 3.4 Predictive Modeling Architecture

An 80/20 stratified split was created to preserve target class proportions:
- **Baseline Model:** Logistic Regression with `class_weight='balanced'` and `StandardScaler` numeric scaling.
- **Ensemble Model:** Random Forest Classifier (300 estimators, `class_weight='balanced'`).
- **Evaluation Criteria:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 4. Key Findings & KPI Results

### 4.1 Overall Churn Benchmark
- Total Customers: **10,000**
- Total Churned: **2,037**
- **Overall Churn Rate: 20.37%** (approximately 1 in every 5 customers leaves the institution).

### 4.2 Geographic Segmentation & Risk Index

| Geography | Total Customers | Churned | Churn Rate | Geographic Risk Index |
|-----------|-----------------|---------|------------|-----------------------|
| **Germany** | 2,509 | 814 | **32.44%** | **1.59 (High Risk)** |
| **Spain** | 2,477 | 413 | **16.67%** | **0.82 (Low Risk)** |
| **France** | 5,014 | 810 | **16.15%** | **0.79 (Low Risk)** |

> **Key Insight:** Customers in Germany churn at double the frequency of customers in France and Spain. Germany accounts for 39.96% of all churned accounts despite representing only 25.09% of the customer base.

### 4.3 Age Group Disparity

| Age Group | Customers | Churned | Churn Rate |
|-----------|-----------|---------|------------|
| `<30` | 1,668 | 126 | **7.55%** |
| `30–45` | 6,367 | 884 | **13.88%** |
| `46–60` | 2,234 | 967 | **43.29%** |
| `60+` | 731 | 186 | **25.44%** |

> **Key Insight:** Churn accelerates steeply past age 45. The 46–60 cohort experiences a staggering 43.29% churn rate.

### 4.4 Financial Profile & High-Value Customer Risk

| Balance Segment | Customers | Churned | Churn Rate |
|-----------------|-----------|---------|------------|
| Zero-balance | 3,617 | 500 | 13.82% |
| Low-balance | 3,191 | 749 | 23.47% |
| High-balance | 3,192 | 788 | **24.69%** |

- **High-Value Churn Ratio:** **24.69%**
- Mean balance of churned customers: **€91,108.54**
- Mean balance of retained customers: **€72,745.30**

> **Key Insight:** Retained customers carry lower average balances than churners. Customers leaving the bank are higher-wealth clients, representing major balance sheet and capital liquidity risk.

### 4.5 Engagement & Activity Analysis

| Member Activity | Customers | Churned | Churn Rate |
|-----------------|-----------|---------|------------|
| Inactive Member (`0`) | 4,849 | 1,302 | **26.85%** |
| Active Member (`1`) | 5,151 | 735 | **14.27%** |

- **Engagement Drop Indicator:** **12.58 percentage points** (Inactive churn is almost double active churn).

### 4.6 Gender Differences
- **Female Churn Rate:** **25.07%** (1,139 / 4,543)
- **Male Churn Rate:** **16.46%** (898 / 5,457)
- Female customers exhibit an 8.61 percentage point higher propensity to exit.

---

## 5. Machine Learning Evaluation

| Metric | Logistic Regression | Random Forest Classifier |
|--------|---------------------|--------------------------|
| **Accuracy** | 70.85% | **85.90%** |
| **Precision** | 38.62% | **76.24%** |
| **Recall** | **71.25%** | 47.91% |
| **F1-Score** | 50.09% | **58.85%** |
| **ROC-AUC** | 0.771 | **0.854** |

- **Model Choice:** The **Random Forest Classifier** is selected as the primary production model due to its high precision (76.24%) and superior area under the ROC curve (0.854), minimizing costly false-positive retention offers.
- **Top Feature Drivers:** `Age`, `NumOfProducts`, `IsActiveMember`, `Balance`, and `Geography_Germany`.

---

## 6. Strategic Recommendations

1. **German Regional Intervention Plan:**
   Conduct targeted customer exit research specifically in Germany to determine if fees, digital product performance, or local fintech competitors (e.g., N26) are driving departures.
2. **Wealth Management & Premier Retention (Ages 45–60):**
   Deploy dedicated relationship managers and attractive term-deposit yields to accounts over €100,000 held by mature clients.
3. **Automated Inactivity Triggers:**
   Establish CRM alerts that flag accounts displaying no transaction activity over 60 days, sending proactive re-engagement incentives before churn takes place.
4. **Product Diversification Strategy:**
   Customers with a single product churn at significantly higher rates; incentivize holding 2 complementary products (e.g., combining checking with an investment/savings wallet).

---

## 7. Conclusion

Customer churn in retail banking is predictable through behavioral and demographic segmentation. By focusing intervention resources on German accounts, mature clients aged 46–60, and inactive high-balance savers, the bank can optimize retention spend and safeguard millions of euros in customer lifetime value.


# 🏦 Customer Segmentation & Churn Pattern Analytics in European Banking

An end-to-end data analytics and machine learning project investigating customer retention, regional exposure, and high-value capital flight across France, Germany, and Spain.

---

## 📌 Project Overview
- **Dataset:** 10,000 European retail banking accounts across France, Spain, and Germany.
- **Overall Churn Rate:** 20.37% (2,037 churned accounts out of 10,000).
- **Core Technology:** Python, Pandas, Scikit-learn, Matplotlib, Streamlit.

---

## 📊 Key Analytical Findings & KPIs

| KPI / Dimension | Value | Strategic Implication |
|---|---|---|
| **Overall Churn Rate** | **20.37%** | Approximately 1 in 5 customers exits the bank |
| **High-Value Churn Ratio** | **24.69%** | Upper-tier balance accounts leave at an elevated rate (€91.1k avg vs €72.7k retained) |
| **Germany Risk Index** | **1.59 (32.44% churn)** | Double the churn rate of France (16.15%) and Spain (16.67%) |
| **Age 46–60 Cohort Churn** | **43.29%** | Severe attrition in primary savings and retirement demographic |
| **Engagement Drop Indicator** | **12.58 pp** | Inactive accounts churn at 26.85% vs 14.27% for active accounts |
| **Gender Disparity** | **25.07% F vs 16.46% M** | Female account holders exhibit 8.61 percentage point higher attrition |

---

## 🤖 Predictive Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 70.85% | 38.62% | **71.25%** | 50.09% | 0.771 |
| **Random Forest (Selected)** | **85.90%** | **76.24%** | 47.91% | **58.85%** | **0.854** |

**Selected Model:** Random Forest Classifier minimizes false-positive interventions while delivering strong predictive power on class-imbalanced data.

---

## 📁 Repository Structure

```
european churn/
├── 01_churn_analysis_and_ml ( unified mentor ).ipynb  # Comprehensive EDA, Segmentation & ML
├── app.py                                             # Interactive Streamlit Web Application
├── European_Bank.csv                                  # Raw bank customer dataset
├── european_bank_cleaned.csv                          # Validated dataset with engineered segments
├── Research_Paper_European_Banking_Churn.md           # Full academic research paper
├── Executive_Summary_European_Banking_Churn.md        # Government & regulatory policy brief
├── requirements.txt                                   # Python dependencies
└── README.md                                          # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open `01_churn_analysis_and_ml ( unified mentor ).ipynb` in VS Code / Antigravity / JupyterLab and execute all cells.

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.


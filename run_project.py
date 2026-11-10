"""
run_project.py  —  One-click project runner for European Bank Churn Analytics
Run this file and it will:
  1. Check / install all required packages
  2. Execute the full analysis and save european_bank_cleaned.csv
  3. Launch the Streamlit dashboard
"""

import subprocess
import sys
import os

# ── Set working directory to the project folder ──────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
print(f"Working directory: {PROJECT_DIR}\n")

# ── 1. Install required packages ─────────────────────────────────────────────
REQUIRED = ["streamlit", "pandas", "numpy", "matplotlib", "scikit-learn"]

print("=" * 60)
print("STEP 1: Checking required packages...")
print("=" * 60)

for pkg in REQUIRED:
    import_name = "sklearn" if pkg == "scikit-learn" else pkg
    try:
        __import__(import_name)
        print(f"  ✅ {pkg} — already installed")
    except ImportError:
        print(f"  ⬇️  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  ✅ {pkg} — installed")

print()

# ── 2. Run the full analysis ──────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves charts to files
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

print("=" * 60)
print("STEP 2: Running full churn analysis...")
print("=" * 60)

# --- Load ---
DATA_PATH = "European_Bank.csv"
if not os.path.exists(DATA_PATH):
    # Try alternate name
    for alt in ["European_Bank(1).csv", "european_bank.csv"]:
        if os.path.exists(alt):
            DATA_PATH = alt
            break
    else:
        print("❌ ERROR: Dataset CSV not found in project folder!")
        print("   Please download it from Google Drive and place it here:")
        print(f"   {PROJECT_DIR}")
        sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  ✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# --- Validation ---
assert df.isnull().sum().sum() == 0, "Missing values found!"
assert df.duplicated().sum() == 0, "Duplicate rows found!"
print(f"  ✅ Validation passed — no nulls, no duplicates")
print(f"  ✅ Churn distribution:")
print(f"     Retained: {(df['Exited']==0).sum():,}  |  Churned: {(df['Exited']==1).sum():,}")

# --- Segmentation ---
df_clean = df.copy()

df_clean["AgeGroup"] = pd.cut(
    df_clean["Age"],
    bins=[-np.inf, 29, 45, 60, np.inf],
    labels=["<30", "30–45", "46–60", "60+"]
)
df_clean["CreditScoreBand"] = pd.cut(
    df_clean["CreditScore"],
    bins=[-np.inf, 579, 669, np.inf],
    labels=["Low", "Medium", "High"]
)
df_clean["TenureGroup"] = pd.cut(
    df_clean["Tenure"],
    bins=[-np.inf, 2, 6, np.inf],
    labels=["New", "Mid-term", "Long-term"]
)
med = df_clean.loc[df_clean["Balance"] > 0, "Balance"].median()
df_clean["BalanceSegment"] = np.select(
    [df_clean["Balance"].eq(0),
     df_clean["Balance"].gt(0) & df_clean["Balance"].le(med),
     df_clean["Balance"].gt(med)],
    ["Zero-balance", "Low-balance", "High-balance"],
    default="Unknown"
)
print(f"  ✅ Segmentation columns created")

# --- KPIs ---
overall_churn = df_clean["Exited"].mean() * 100
hv = df_clean[df_clean["BalanceSegment"] == "High-balance"]
hv_churn = hv["Exited"].mean() * 100
inactive_churn = df_clean.loc[df_clean["IsActiveMember"]==0, "Exited"].mean() * 100
active_churn   = df_clean.loc[df_clean["IsActiveMember"]==1, "Exited"].mean() * 100
engagement_drop = inactive_churn - active_churn

def seg_rate(data, col):
    r = (data.groupby(col, observed=False)["Exited"]
         .agg(Customers="count", Churned="sum", ChurnRate="mean")
         .assign(ChurnRate=lambda x: x["ChurnRate"]*100))
    return r

geo = seg_rate(df_clean, "Geography")
geo["GeographicRiskIndex"] = geo["ChurnRate"] / overall_churn

print()
print("  📊 KEY PERFORMANCE INDICATORS")
print(f"  {'Overall Churn Rate':<30} {overall_churn:.2f}%")
print(f"  {'High-Value Churn Ratio':<30} {hv_churn:.2f}%")
print(f"  {'Inactive Member Churn Rate':<30} {inactive_churn:.2f}%")
print(f"  {'Active Member Churn Rate':<30} {active_churn:.2f}%")
print(f"  {'Engagement Drop Indicator':<30} {engagement_drop:.2f} pp")
print()
print("  🌍 GEOGRAPHIC RISK INDEX")
for country, row in geo.iterrows():
    risk_label = "⚠️ HIGH" if row["GeographicRiskIndex"] > 1 else "✅ LOW"
    print(f"  {country:<10} Churn: {row['ChurnRate']:.1f}%  Risk Index: {row['GeographicRiskIndex']:.2f}  {risk_label}")

# --- Segment analysis ---
print()
print("  📋 CHURN BY SEGMENT")
for label, col in [("Age Group", "AgeGroup"), ("Balance", "BalanceSegment"), ("Gender", "Gender")]:
    r = seg_rate(df_clean, col).sort_values("ChurnRate", ascending=False)
    print(f"\n  {label}:")
    for seg, row in r.iterrows():
        bar = "█" * int(row["ChurnRate"] / 2)
        print(f"    {str(seg):<15} {row['ChurnRate']:5.1f}%  {bar}")

# --- Save charts ---
os.makedirs("charts", exist_ok=True)
segments = [
    ("Geography",      "Churn Rate by Geography",      "#5B8DB8"),
    ("AgeGroup",       "Churn Rate by Age Group",       "#F4A261"),
    ("TenureGroup",    "Churn Rate by Tenure Group",    "#6D9B78"),
    ("CreditScoreBand","Churn Rate by Credit Score",    "#9B59B6"),
    ("BalanceSegment", "Churn Rate by Balance Segment", "#1ABC9C"),
    ("Gender",         "Churn Rate by Gender",          "#E91E63"),
]
for col, title, color in segments:
    rates = df_clean.groupby(col, observed=False)["Exited"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates.index.astype(str), rates.values, color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(ax.patches, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fname = f"charts/{col.lower()}_churn.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)

print(f"\n  ✅ Charts saved to: charts/")

# --- ML Models ---
print()
print("  🤖 Training ML models...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

model_features = ["CreditScore","Geography","Gender","Age","Tenure",
                  "Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
X = df_clean[model_features]
y = df_clean["Exited"]
cat_feats = ["Geography", "Gender"]
num_feats = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
             "HasCrCard","IsActiveMember","EstimatedSalary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_lr = ColumnTransformer([("num", StandardScaler(), num_feats),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)])
lr = Pipeline([("pre", pre_lr),
               ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:,1]

pre_rf = ColumnTransformer([("num", "passthrough", num_feats),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)])
rf = Pipeline([("pre", pre_rf),
               ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print()
print("  MODEL RESULTS:")
print(f"  {'Model':<22} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
print(f"  {'-'*50}")
for name, pred, prob in [("Logistic Regression", lr_pred, lr_prob), ("Random Forest", rf_pred, rf_prob)]:
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)
    print(f"  {name:<22} {acc:>8.1%} {f1:>8.3f} {auc:>9.3f}")

best_model = "Random Forest" if f1_score(y_test, rf_pred) >= f1_score(y_test, lr_pred) else "Logistic Regression"
print(f"\n  🏆 Best model by F1: {best_model}")

# --- Save cleaned CSV ---
OUTPUT = "european_bank_cleaned.csv"
df_clean.to_csv(OUTPUT, index=False)
print(f"  ✅ Cleaned dataset saved → {OUTPUT}")

print()
print("=" * 60)
print("✅ STEP 2 COMPLETE — Analysis finished!")
print("=" * 60)

# --- Launch Streamlit ---
print()
print("=" * 60)
print("STEP 3: Launching Streamlit Dashboard...")
print("  → Opening http://localhost:8501 in your browser")
print("  → Press Ctrl+C in terminal to stop the dashboard")
print("=" * 60)
print()

subprocess.run([sys.executable, "-m", "streamlit", "run",
                os.path.join(PROJECT_DIR, "app.py"),
                "--server.headless", "false"],
               cwd=PROJECT_DIR)


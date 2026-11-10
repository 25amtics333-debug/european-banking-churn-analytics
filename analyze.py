"""
QUICK ANALYSIS — Run this file directly in Antigravity with the ▶ Run button
(Right-click this file in Explorer → Run Python File)
"""

import os, sys, subprocess

# Fix working directory regardless of how the script is launched
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("📂 Working dir:", os.getcwd())

# Install packages silently if missing
def ensure(pkg, import_as=None):
    name = import_as or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for p, i in [("streamlit",None),("pandas",None),("numpy",None),
             ("matplotlib",None),("scikit-learn","sklearn")]:
    ensure(p, i)
print("✅ All packages ready\n")

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mtick

# ── Find CSV ──────────────────────────────────────────────────────────────────
csv = next((f for f in ["European_Bank.csv","European_Bank(1).csv"]
            if os.path.exists(f)), None)
if csv is None:
    print("❌ CSV not found — make sure European_Bank.csv is in this folder")
    sys.exit(1)

df = pd.read_csv(csv)
print(f"✅ Loaded {csv}: {df.shape[0]:,} rows")

# ── Segments ──────────────────────────────────────────────────────────────────
df["AgeGroup"] = pd.cut(df["Age"], [-np.inf,29,45,60,np.inf],
                         labels=["<30","30–45","46–60","60+"])
df["CreditScoreBand"] = pd.cut(df["CreditScore"], [-np.inf,579,669,np.inf],
                                labels=["Low","Medium","High"])
df["TenureGroup"] = pd.cut(df["Tenure"], [-np.inf,2,6,np.inf],
                            labels=["New","Mid-term","Long-term"])
med = df.loc[df["Balance"]>0,"Balance"].median()
df["BalanceSegment"] = np.select(
    [df["Balance"].eq(0), df["Balance"].le(med) & df["Balance"].gt(0), df["Balance"].gt(med)],
    ["Zero-balance","Low-balance","High-balance"], "Unknown")
print("✅ Segmentation done")

# ── KPIs ──────────────────────────────────────────────────────────────────────
ocr = df["Exited"].mean()*100
hvc = df[df["BalanceSegment"]=="High-balance"]["Exited"].mean()*100
ic  = df[df["IsActiveMember"]==0]["Exited"].mean()*100
ac  = df[df["IsActiveMember"]==1]["Exited"].mean()*100

geo = df.groupby("Geography")["Exited"].agg(ChurnRate="mean")
geo["ChurnRate"]*=100
geo["RiskIndex"] = geo["ChurnRate"]/ocr

print(f"\n📊 KPIs")
print(f"   Overall Churn Rate        : {ocr:.2f}%")
print(f"   High-Value Churn Ratio    : {hvc:.2f}%")
print(f"   Inactive Member Churn     : {ic:.2f}%")
print(f"   Active Member Churn       : {ac:.2f}%")
print(f"   Engagement Drop Indicator : {ic-ac:.2f} pp")
print(f"\n🌍 Geographic Risk Index")
print(geo.to_string())

# ── Charts ────────────────────────────────────────────────────────────────────
os.makedirs("charts", exist_ok=True)
for col, title, color in [
    ("Geography","Churn by Geography","#5B8DB8"),
    ("AgeGroup","Churn by Age Group","#F4A261"),
    ("TenureGroup","Churn by Tenure","#6D9B78"),
    ("CreditScoreBand","Churn by Credit Score","#9B59B6"),
    ("BalanceSegment","Churn by Balance","#1ABC9C"),
    ("Gender","Churn by Gender","#E91E63"),
]:
    r = df.groupby(col, observed=False)["Exited"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(r.index.astype(str), r.values, color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for b,v in zip(ax.patches, r.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(f"charts/{col.lower().replace(' ','_')}_churn.png", dpi=120)
    plt.close(fig)
print("✅ Charts saved to charts/")

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

feats = ["CreditScore","Geography","Gender","Age","Tenure",
         "Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
num_f = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
         "HasCrCard","IsActiveMember","EstimatedSalary"]
cat_f = ["Geography","Gender"]

X, y = df[feats], df["Exited"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_lr = ColumnTransformer([("n",StandardScaler(),num_f),("c",OneHotEncoder(handle_unknown="ignore"),cat_f)])
lr = Pipeline([("p",pre_lr),("m",LogisticRegression(max_iter=1000,class_weight="balanced"))])
lr.fit(Xtr,ytr); lp=lr.predict(Xte); lb=lr.predict_proba(Xte)[:,1]

pre_rf = ColumnTransformer([("n","passthrough",num_f),("c",OneHotEncoder(handle_unknown="ignore"),cat_f)])
rf = Pipeline([("p",pre_rf),("m",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
rf.fit(Xtr,ytr); rp=rf.predict(Xte); rb=rf.predict_proba(Xte)[:,1]

print("\n🤖 ML Model Results:")
print(f"   {'Model':<22} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
print(f"   {'-'*52}")
for n,p,b in [("Logistic Regression",lp,lb),("Random Forest",rp,rb)]:
    print(f"   {n:<22} {accuracy_score(yte,p):>8.1%} {f1_score(yte,p):>8.3f} {roc_auc_score(yte,b):>9.3f}")

best = "Random Forest" if f1_score(yte,rp)>=f1_score(yte,lp) else "Logistic Regression"
print(f"\n   🏆 Best model: {best}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("european_bank_cleaned.csv", index=False)
print("\n✅ european_bank_cleaned.csv saved!")

print("\n" + "="*60)
print("ALL DONE! Now run:  streamlit run app.py")
print("="*60)


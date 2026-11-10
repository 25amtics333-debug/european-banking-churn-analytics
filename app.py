"""
Streamlit Dashboard — Customer Segmentation & Churn Pattern Analytics
European Banking Dataset  |  Unified Mentor Project
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("european_bank_cleaned.csv")
    except FileNotFoundError:
        # Fall back to the raw dataset and add segments on the fly
        df = pd.read_csv("European_Bank.csv")
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 29, 45, 60, np.inf],
            labels=["<30", "30–45", "46–60", "60+"],
        )
        df["CreditScoreBand"] = pd.cut(
            df["CreditScore"],
            bins=[-np.inf, 579, 669, np.inf],
            labels=["Low", "Medium", "High"],
        )
        df["TenureGroup"] = pd.cut(
            df["Tenure"],
            bins=[-np.inf, 2, 6, np.inf],
            labels=["New", "Mid-term", "Long-term"],
        )
        med = df.loc[df["Balance"] > 0, "Balance"].median()
        df["BalanceSegment"] = np.select(
            [df["Balance"].eq(0),
             df["Balance"].gt(0) & df["Balance"].le(med),
             df["Balance"].gt(med)],
            ["Zero-balance", "Low-balance", "High-balance"],
            default="Unknown",
        )
    return df


df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔎 Filters")

# Geography
geo_options = ["All"] + sorted(df_full["Geography"].dropna().unique().tolist())
selected_geo = st.sidebar.multiselect("Geography", geo_options[1:], default=geo_options[1:])

# Gender
gender_options = sorted(df_full["Gender"].dropna().unique().tolist())
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

# Age Group
age_options = ["<30", "30–45", "46–60", "60+"]
selected_age = st.sidebar.multiselect("Age Group", age_options, default=age_options)

# Active Member
active_filter = st.sidebar.radio("Member Status", ["All", "Active only", "Inactive only"])

# Apply filters
df = df_full.copy()
if selected_geo:
    df = df[df["Geography"].isin(selected_geo)]
if selected_gender:
    df = df[df["Gender"].isin(selected_gender)]
if selected_age:
    df = df[df["AgeGroup"].isin(selected_age)]
if active_filter == "Active only":
    df = df[df["IsActiveMember"] == 1]
elif active_filter == "Inactive only":
    df = df[df["IsActiveMember"] == 0]

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.caption("European Banking Dataset  |  Unified Mentor Project")
st.divider()

if df.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
overall_churn = df["Exited"].mean() * 100
total_customers = len(df)
total_churned = df["Exited"].sum()

high_value = df[df["BalanceSegment"] == "High-balance"]
hv_churn = high_value["Exited"].mean() * 100 if len(high_value) > 0 else 0.0

inactive_churn = df.loc[df["IsActiveMember"] == 0, "Exited"].mean() * 100
active_churn_r  = df.loc[df["IsActiveMember"] == 1, "Exited"].mean() * 100
engagement_drop = inactive_churn - active_churn_r

st.subheader("📊 Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Churn Rate", f"{overall_churn:.1f}%")
k2.metric("Total Customers", f"{total_customers:,}")
k3.metric("Total Churned", f"{int(total_churned):,}")
k4.metric("High-Value Churn Ratio", f"{hv_churn:.1f}%")
k5.metric("Engagement Drop Indicator", f"{engagement_drop:.1f} pp")

st.divider()

# ── Helper ────────────────────────────────────────────────────────────────────
def seg_churn(data, col):
    return (
        data.groupby(col, observed=False)["Exited"]
        .agg(Customers="count", Churned="sum", ChurnRate="mean")
        .assign(ChurnRate=lambda x: x["ChurnRate"] * 100)
        .sort_values("ChurnRate", ascending=False)
        .reset_index()
    )


def bar_chart(data, x_col, title, color="#E05C5C"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data[x_col].astype(str), data["ChurnRate"], color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(ax.patches, data["ChurnRate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Overall Churn Summary
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Overall Churn Summary")
col_a, col_b = st.columns([1, 2])

with col_a:
    retained = total_customers - total_churned
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(
        [retained, total_churned],
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#E05C5C"],
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax_pie.set_title("Churn vs Retained", fontweight="bold")
    st.pyplot(fig_pie)
    plt.close(fig_pie)

with col_b:
    profile_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                    "EstimatedSalary", "IsActiveMember", "HasCrCard"]
    profile = df.groupby("Exited")[profile_cols].mean().T
    profile.columns = ["Retained", "Churned"]
    profile.index.name = "Feature"
    st.write("**Average Profile: Churned vs Retained**")
    st.dataframe(profile.style.format("{:,.2f}"))

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Geography-wise Churn
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🌍 Geography-wise Churn Visualization")
geo_data = seg_churn(df, "Geography")
geo_data["GeographicRiskIndex"] = geo_data["ChurnRate"] / overall_churn

col1, col2 = st.columns(2)
with col1:
    st.pyplot(bar_chart(geo_data, "Geography", "Churn Rate by Geography", "#5B8DB8"))
    plt.close()

with col2:
    st.write("**Geographic Risk Index** (>1 = higher than average risk)")
    st.dataframe(
        geo_data[["Geography", "Customers", "Churned", "ChurnRate", "GeographicRiskIndex"]]
        .style.format({"ChurnRate": "{:.1f}%", "GeographicRiskIndex": "{:.2f}"})
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Age & Tenure Churn Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Age & Tenure Churn Comparison")
age_data = seg_churn(df, "AgeGroup")
tenure_data = seg_churn(df, "TenureGroup")

col3, col4 = st.columns(2)
with col3:
    st.pyplot(bar_chart(age_data, "AgeGroup", "Churn Rate by Age Group", "#F4A261"))
    plt.close()

with col4:
    st.pyplot(bar_chart(tenure_data, "TenureGroup", "Churn Rate by Tenure Group", "#6D9B78"))
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Credit Score & Balance Segments
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💳 Credit Score & Balance Segment Churn")
cs_data = seg_churn(df, "CreditScoreBand")
bal_data = seg_churn(df, "BalanceSegment")

col5, col6 = st.columns(2)
with col5:
    st.pyplot(bar_chart(cs_data, "CreditScoreBand", "Churn Rate by Credit Score Band", "#9B59B6"))
    plt.close()

with col6:
    st.pyplot(bar_chart(bal_data, "BalanceSegment", "Churn Rate by Balance Segment", "#1ABC9C"))
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — High-Value Customer Churn Explorer
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💰 High-Value Customer Churn Explorer")

hv_df = df[df["BalanceSegment"] == "High-balance"].copy()
nonhv_df = df[df["BalanceSegment"] != "High-balance"].copy()

hvc1, hvc2, hvc3 = st.columns(3)
hvc1.metric("High-Value Customers", f"{len(hv_df):,}")
hvc2.metric("High-Value Churned", f"{int(hv_df['Exited'].sum()):,}")
hvc3.metric("High-Value Churn Rate", f"{hv_df['Exited'].mean()*100:.1f}%")

col7, col8 = st.columns(2)
with col7:
    # High-value churn by geography
    hv_geo = seg_churn(hv_df, "Geography")
    st.pyplot(bar_chart(hv_geo, "Geography", "High-Value Churn by Geography", "#E74C3C"))
    plt.close()

with col8:
    # High-value churn by age group
    hv_age = seg_churn(hv_df, "AgeGroup")
    st.pyplot(bar_chart(hv_age, "AgeGroup", "High-Value Churn by Age Group", "#E67E22"))
    plt.close()

# Balance vs Salary scatter
st.write("**High-Value Customers: Balance vs Estimated Salary (by Churn status)**")
fig_sc, ax_sc = plt.subplots(figsize=(10, 4))
colors_map = {0: "#4CAF50", 1: "#E05C5C"}
for status, label, marker in [(0, "Retained", "o"), (1, "Churned", "X")]:
    sub = hv_df[hv_df["Exited"] == status]
    ax_sc.scatter(
        sub["EstimatedSalary"], sub["Balance"],
        c=colors_map[status], label=label,
        alpha=0.5, s=20, marker=marker,
    )
ax_sc.set_xlabel("Estimated Salary")
ax_sc.set_ylabel("Balance")
ax_sc.set_title("High-Value Customers: Salary vs Balance")
ax_sc.legend()
ax_sc.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax_sc.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
fig_sc.tight_layout()
st.pyplot(fig_sc)
plt.close(fig_sc)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Gender & Engagement Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚡ Gender & Engagement Analysis")
gender_data = seg_churn(df, "Gender")
engagement_data = (
    df.groupby("IsActiveMember")["Exited"]
    .agg(Customers="count", Churned="sum", ChurnRate="mean")
    .assign(
        ChurnRate=lambda x: x["ChurnRate"] * 100,
        Status=lambda x: x.index.map({0: "Inactive", 1: "Active"}),
    )
    .reset_index()
)

col9, col10 = st.columns(2)
with col9:
    st.pyplot(bar_chart(gender_data, "Gender", "Churn Rate by Gender", "#E91E63"))
    plt.close()

with col10:
    fig_eng, ax_eng = plt.subplots(figsize=(6, 4))
    ax_eng.bar(["Inactive", "Active"],
               [inactive_churn, active_churn_r],
               color=["#E05C5C", "#4CAF50"])
    ax_eng.set_title("Active vs Inactive Member Churn Rate", fontweight="bold")
    ax_eng.set_ylabel("Churn Rate (%)")
    ax_eng.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(ax_eng.patches, [inactive_churn, active_churn_r]):
        ax_eng.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    fig_eng.tight_layout()
    st.pyplot(fig_eng)
    plt.close(fig_eng)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Geography × Age Interaction (Drill-down)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔬 Drill-Down: Geography × Age Interaction")

geo_age = (
    df.groupby(["Geography", "AgeGroup"], observed=False)["Exited"]
    .agg(Customers="count", Churned="sum", ChurnRate="mean")
    .assign(ChurnRate=lambda x: x["ChurnRate"] * 100)
    .reset_index()
)

pivot = geo_age.pivot(index="Geography", columns="AgeGroup", values="ChurnRate").fillna(0)

fig_heat, ax_heat = plt.subplots(figsize=(8, 3))
im = ax_heat.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels(pivot.columns, fontsize=11)
ax_heat.set_yticks(range(len(pivot.index)))
ax_heat.set_yticklabels(pivot.index, fontsize=11)
ax_heat.set_title("Churn Rate Heatmap: Geography × Age Group", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax_heat, format="%.1f%%")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax_heat.text(j, i, f"{pivot.values[i, j]:.1f}%",
                     ha="center", va="center", fontsize=9,
                     color="black" if pivot.values[i, j] < 40 else "white")
fig_heat.tight_layout()
st.pyplot(fig_heat)
plt.close(fig_heat)

# Raw table
st.write("**Segment-Level Detail Table**")
detail = df.copy()
detail["Exited_Label"] = detail["Exited"].map({0: "Retained", 1: "Churned"})
st.dataframe(
    geo_age.sort_values("ChurnRate", ascending=False)
    .style.format({"ChurnRate": "{:.1f}%"})
)

st.divider()
st.caption("Dashboard built with Streamlit  |  Unified Mentor — Customer Churn Analytics Project")


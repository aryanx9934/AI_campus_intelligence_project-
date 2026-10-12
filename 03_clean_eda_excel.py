"""
03_clean_eda_excel.py
Loads raw data via SQL, cleans it with Pandas, runs EDA, and exports
a multi-sheet Excel workbook (raw, cleaned, summary, watchlist) --
the "Excel reporting" layer of the workflow.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
out_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(out_dir, "campus_intelligence.db")

# ---- 1. Extract via SQL (this mirrors query #2 in the .sql file) ----
conn = sqlite3.connect(db_path)
raw_df = pd.read_sql("""
    SELECT * FROM students_raw
""", conn)
conn.close()

print(f"Loaded {len(raw_df)} raw rows from SQLite.")

# ---- 2. Clean with Pandas ----
df = raw_df.copy()

# Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates(subset=["student_id"])
print(f"Dropped {before - len(df)} duplicate rows.")

# Fix bad data entry (negative internship_count is impossible)
bad_rows = (df["internship_count"] < 0).sum()
df = df[df["internship_count"] >= 0]
print(f"Removed {bad_rows} rows with invalid internship_count.")

# Impute missing communication_score with department-wise median
df["communication_score"] = df.groupby("department")["communication_score"] \
    .transform(lambda s: s.fillna(s.median()))

# Feature engineering
df["at_risk_label"] = df["at_risk"].map({1: "At Risk", 0: "On Track"})
df["placed_label"] = df["placed"].map({1: "Placed", 0: "Not Placed"})
df["performance_index"] = (
    0.4 * df["cgpa"] * 10
    + 0.2 * df["coding_score"]
    + 0.2 * df["aptitude_score"]
    + 0.2 * df["communication_score"]
).round(1)

cleaned_csv_path = os.path.join(out_dir, "cleaned_student_data.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"Saved cleaned data -> {cleaned_csv_path}")

# ---- 3. EDA charts (saved as images for the report / dashboard) ----
charts_dir = os.path.join(out_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["cgpa"], bins=20, kde=True, color="#2c3e91", ax=ax)
ax.set_title("CGPA Distribution")
fig.tight_layout()
fig.savefig(os.path.join(charts_dir, "cgpa_distribution.png"), dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
dept_placement = df.groupby("department")["placed"].mean().sort_values(ascending=False) * 100
sns.barplot(x=dept_placement.index, y=dept_placement.values, color="#2c3e91", ax=ax)
ax.set_ylabel("Placement Rate (%)")
ax.set_title("Placement Rate by Department")
fig.tight_layout()
fig.savefig(os.path.join(charts_dir, "placement_by_department.png"), dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="at_risk_label", y="attendance_pct", palette=["#2c3e91", "#c0392b"], ax=ax)
ax.set_title("Attendance % by Risk Status")
fig.tight_layout()
fig.savefig(os.path.join(charts_dir, "attendance_by_risk.png"), dpi=140)
plt.close(fig)

corr_cols = ["cgpa", "attendance_pct", "backlog_count", "coding_score",
             "aptitude_score", "communication_score", "internship_count", "projects_count"]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
fig.tight_layout()
fig.savefig(os.path.join(charts_dir, "correlation_heatmap.png"), dpi=140)
plt.close(fig)

print(f"Saved 4 EDA charts -> {charts_dir}")

# ---- 4. Multi-sheet Excel export ----
excel_path = os.path.join(out_dir, "Campus_Intelligence_Report.xlsx")

dept_summary = df.groupby("department").agg(
    num_students=("student_id", "count"),
    avg_cgpa=("cgpa", "mean"),
    avg_attendance=("attendance_pct", "mean"),
    at_risk_count=("at_risk", "sum"),
    placement_rate_pct=("placed", "mean"),
).reset_index()
dept_summary["avg_cgpa"] = dept_summary["avg_cgpa"].round(2)
dept_summary["avg_attendance"] = dept_summary["avg_attendance"].round(1)
dept_summary["placement_rate_pct"] = (dept_summary["placement_rate_pct"] * 100).round(1)

watchlist = df[df["at_risk"] == 1][[
    "student_id", "department", "semester", "cgpa",
    "attendance_pct", "backlog_count", "performance_index"
]].sort_values("performance_index")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
    df.to_excel(writer, sheet_name="Cleaned Data", index=False)
    dept_summary.to_excel(writer, sheet_name="Department Summary", index=False)
    watchlist.to_excel(writer, sheet_name="At-Risk Watchlist", index=False)

print(f"Saved multi-sheet Excel report -> {excel_path}")

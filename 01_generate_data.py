"""
01_generate_data.py
Generates a realistic synthetic student records dataset for the
AI-Powered Campus Intelligence Dashboard project, and loads it into
a SQLite database (simulating the "SQL extraction" layer of the workflow).
"""
import numpy as np
import pandas as pd
import sqlite3
import os

np.random.seed(42)

N = 600  # number of student records

departments = ["CSE", "ECE", "Mechanical", "Civil", "Electrical", "IT"]
dept_probs = [0.30, 0.20, 0.15, 0.10, 0.10, 0.15]

student_id = [f"STU{str(i).zfill(4)}" for i in range(1, N + 1)]
department = np.random.choice(departments, size=N, p=dept_probs)
semester = np.random.choice([4, 5, 6, 7, 8], size=N)

# Academic performance (correlated across semesters, with some noise)
base_ability = np.random.normal(7.0, 1.2, N).clip(3, 10)
attendance_pct = (base_ability * 5 + 45 + np.random.normal(0, 6, N)).clip(40, 100)
cgpa = (base_ability + np.random.normal(0, 0.5, N)).clip(4.0, 10.0)
backlog_count = np.random.poisson(lam=(10 - cgpa).clip(0, None) * 0.6).clip(0, 8)

# Skills / extracurricular
coding_score = (base_ability * 9 + np.random.normal(0, 10, N)).clip(0, 100)
aptitude_score = (base_ability * 8.5 + np.random.normal(0, 12, N)).clip(0, 100)
communication_score = np.random.normal(65, 15, N).clip(0, 100)
extracurricular_score = np.random.normal(50, 20, N).clip(0, 100)
internship_count = np.random.poisson(lam=(base_ability / 4)).clip(0, 4)
projects_count = np.random.poisson(lam=(base_ability / 3)).clip(0, 6)

# Socioeconomic / support factors (used as fairness-relevant but non-punitive context)
family_income_bracket = np.random.choice(["Low", "Medium", "High"], size=N, p=[0.35, 0.45, 0.20])
first_gen_learner = np.random.choice([0, 1], size=N, p=[0.6, 0.4])

# --- Derived label 1: At-risk student (dropout / academic risk) ---
risk_score = (
    -1.6
    - 0.55 * (cgpa - 7)
    - 0.045 * (attendance_pct - 75)
    + 0.35 * backlog_count
    - 0.015 * (coding_score - 60)
    + np.random.normal(0, 0.6, N)
)
at_risk_prob = 1 / (1 + np.exp(-risk_score))
at_risk = (at_risk_prob > 0.5).astype(int)

# --- Derived label 2: Placement outcome (probability of being placed) ---
placement_score = (
    0.2
    + 0.45 * (cgpa - 7)
    + 0.02 * (coding_score - 60)
    + 0.015 * (aptitude_score - 60)
    + 0.01 * (communication_score - 60)
    + 0.3 * internship_count
    + 0.15 * projects_count
    - 0.3 * backlog_count
    + np.random.normal(0, 0.8, N)
)
placement_prob = 1 / (1 + np.exp(-placement_score))
placed = (placement_prob > 0.5).astype(int)

df = pd.DataFrame({
    "student_id": student_id,
    "department": department,
    "semester": semester,
    "attendance_pct": attendance_pct.round(1),
    "cgpa": cgpa.round(2),
    "backlog_count": backlog_count,
    "coding_score": coding_score.round(1),
    "aptitude_score": aptitude_score.round(1),
    "communication_score": communication_score.round(1),
    "extracurricular_score": extracurricular_score.round(1),
    "internship_count": internship_count,
    "projects_count": projects_count,
    "family_income_bracket": family_income_bracket,
    "first_gen_learner": first_gen_learner,
    "at_risk": at_risk,
    "placed": placed,
})

# Inject some realistic messiness for the cleaning step to handle
missing_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "communication_score"] = np.nan
dup_rows = df.sample(5, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)
neg_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[neg_idx, "internship_count"] = -1  # bad data entry

out_dir = os.path.dirname(os.path.abspath(__file__))
raw_csv_path = os.path.join(out_dir, "raw_student_data.csv")
df.to_csv(raw_csv_path, index=False)

# Load into SQLite to simulate the "SQL extraction" layer
db_path = os.path.join(out_dir, "campus_intelligence.db")
conn = sqlite3.connect(db_path)
df.to_sql("students_raw", conn, if_exists="replace", index=False)
conn.close()

print(f"Generated {len(df)} raw records (with intentional messiness).")
print(f"Saved CSV -> {raw_csv_path}")
print(f"Loaded into SQLite -> {db_path} (table: students_raw)")

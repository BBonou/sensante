"""
SenSante - patients_dakar.csv dataset exploration
Lab 1 : Git, Python et Project's structure
"""

import pandas as pd

# ===== LOAD THE DATA =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== FIRST SIGHT =====
print("=" * 50)
print("SENSANTE - Dataset Exploration")
print("=" * 50)

# Dataset dimensions
print(f"\nNumber of patients : {len(df)}")
print(f"Columns number : {df.shape[1]}")
print(f"Columns : {list(df.columns)}")

# Sight of the 5 first lines
print(f"\n--- 5 first patients ---")
print(df.head())

# ===== BASE STATISTICS =====
print(f"\n--- Descriptive statistics ---")
print(df.describe().round(2))

# ===== DISTRIBUTION OF DIAGNOSES =====
print(f"\n--- Distribution of diagnoses ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"   {diag:12s} : {count:3d}, patients ({pct:.1f}%)")

# ===== DISTRIBUTION BY REGION =====
print(f"\n--- Distribution by region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f"   {region:15s} : {count:3d} patients")

# ===== AVERAGE TEMPERATURE PER DIAGNOSTIC =====
print(f"\n--- Average temperature per diagnostic ---")
tmp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in tmp_by_diag.items():
    print(f"   {diag:12s} : {temp:.1f} C")

# ===== NUMBER OF PATIENTS PER GENDER & DIAGNOSTIC=====
print(f"\n--- Number of patients per gender & diagnostic ---")
sexe_diag_count = df.groupby(["sexe", "diagnostic"]).size()
for (sexe, diag), count in sexe_diag_count.items():
    print(f"  {sexe} - {diag:12s} : {count:3d} patients")

print(f"\n{'=' * 50}")
print("Exploration finished")
print("Next lab ; train a model ML")
print(f"{'=' * 50}")
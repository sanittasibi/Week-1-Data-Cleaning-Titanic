# Week 1 - Data Acquisition, Cleaning and Preprocessing
# YUVA Intern - Virtual Data Science with Python Trainee

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Data acquisition
# -----------------------------
df = sns.load_dataset("titanic")

# -----------------------------
# 2. Initial exploration
# -----------------------------
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df.describe())

# -----------------------------
# 3. Missing-value analysis
# -----------------------------
missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100
missing_data = pd.DataFrame({
    "Missing Values": missing_count,
    "Percentage": missing_percentage.round(2)
})
missing_data = missing_data[missing_data["Missing Values"] > 0]
print("\nMissing-value analysis:")
print(missing_data)

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values in Titanic Dataset")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Duplicate and consistency checks
# -----------------------------
duplicates = df[df.duplicated(keep=False)]
print("\nSample duplicate records:")
print(duplicates.head(20))

for column in ["sex", "embarked", "class", "who"]:
    print(f"\n{column} values:")
    print(df[column].value_counts())

# -----------------------------
# 5. Cleaning
# -----------------------------
df_clean = df.copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])
df_clean["embark_town"] = df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0])
df_clean = df_clean.drop(columns=["deck"])

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# -----------------------------
# 6. Outlier analysis using IQR
# -----------------------------
numerical_columns = ["age", "fare", "sibsp", "parch"]

for column in numerical_columns:
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[column] < lower_bound) |
        (df_clean[column] > upper_bound)
    ]

    print(f"\n{column}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Outlier Count: {len(outliers)}")

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df_clean[column])
    plt.title(f"Box Plot of {column}")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Preprocessing / encoding
# -----------------------------
df_processed = df_clean.copy()
df_processed["adult_male"] = df_processed["adult_male"].astype(int)
df_processed["alone"] = df_processed["alone"].astype(int)

df_processed = pd.get_dummies(
    df_processed,
    columns=["sex", "embarked", "class", "who", "embark_town", "alive"],
    drop_first=True,
    dtype=int
)

print("\nProcessed dataset shape:", df_processed.shape)
print(df_processed.head())

# -----------------------------
# 8. Modelling-ready version
# -----------------------------
df_model_ready = df_clean.drop(columns=["alive"])
print("\nModelling-ready shape:", df_model_ready.shape)
print("Modelling-ready columns:", df_model_ready.columns.tolist())

# -----------------------------
# 9. Final validation
# -----------------------------
print("\n========== FINAL VALIDATION ==========")
print("Rows:", df_clean.shape[0])
print("Columns after cleaning:", df_clean.shape[1])
print("Total missing values:", df_clean.isnull().sum().sum())
print("Exact duplicate rows after cleaning:", df_clean.duplicated().sum())

# -----------------------------
# 10. Save output
# -----------------------------
df_processed.to_csv("titanic_cleaned_processed.csv", index=False)
print("\nSaved: titanic_cleaned_processed.csv")

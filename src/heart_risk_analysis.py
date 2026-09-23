# ============================================================
# HEART RISK ANALYTICS AND PREDICTION
# Machine Learning Project
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv("../data/heart_attack_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -----------------------------
# 3. Basic Data Information
# -----------------------------

print("\n===== DATA INFORMATION =====")
print(df.info())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Duplicate rows:", df.duplicated().sum())


# -----------------------------
# 4. Target Variable Analysis
# -----------------------------

print("\n===== HEART ATTACK RISK DISTRIBUTION =====")

print(df["Heart_Attack_Risk"].value_counts())

print("\nRisk percentages:")
print(
    df["Heart_Attack_Risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# 5. Statistical Summary
# -----------------------------

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# -----------------------------
# 6. Categorical Distributions
# -----------------------------

categorical_columns = [
    "Gender",
    "State_Name",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Smoking",
    "Alcohol_Consumption",
    "Physical_Activity",
    "Family_History",
    "Healthcare_Access",
    "Heart_Attack_History",
    "Health_Insurance",
    "Heart_Attack_Risk"
]

print("\n===== CATEGORICAL DISTRIBUTIONS =====")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())


# -----------------------------
# 7. Heart Attack Risk Graph
# -----------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Heart_Attack_Risk"
)

plt.title("Heart Attack Risk Distribution")
plt.xlabel("Heart Attack Risk")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# -----------------------------
# 8. Risk Factor Analysis
# -----------------------------

risk_factors = [
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Smoking",
    "Alcohol_Consumption",
    "Physical_Activity",
    "Family_History",
    "Heart_Attack_History"
]

print("\n===== RISK FACTORS VS HEART ATTACK RISK =====")

for factor in risk_factors:

    print(f"\n===== {factor} =====")

    result = pd.crosstab(
        df[factor],
        df["Heart_Attack_Risk"],
        normalize="index"
    ) * 100

    print(result.round(2))


# -----------------------------
# 9. Risk Factor Visualizations
# -----------------------------

factors = [
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Smoking",
    "Family_History"
]

for factor in factors:

    plt.figure(figsize=(7, 4))

    sns.countplot(
        data=df,
        x=factor,
        hue="Heart_Attack_Risk"
    )

    plt.title(f"{factor} vs Heart Attack Risk")
    plt.xlabel(factor)
    plt.ylabel("Number of Patients")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 10. Numerical Factor Analysis
# -----------------------------

numerical_factors = [
    "Age",
    "Diet_Score",
    "Cholesterol_Level",
    "Triglyceride_Level",
    "LDL_Level",
    "HDL_Level",
    "Systolic_BP",
    "Diastolic_BP",
    "Stress_Level",
    "Emergency_Response_Time",
    "Annual_Income"
]

print("\n===== NUMERICAL FACTORS VS HEART ATTACK RISK =====")

for factor in numerical_factors:

    print(f"\n===== {factor} =====")

    result = (
        df.groupby("Heart_Attack_Risk")[factor]
        .mean()
        .round(2)
    )

    print(result)


# -----------------------------
# 11. Numerical Factor Graphs
# -----------------------------

boxplot_factors = [
    "Age",
    "Cholesterol_Level",
    "LDL_Level",
    "HDL_Level",
    "Systolic_BP"
]

for factor in boxplot_factors:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=df,
        x="Heart_Attack_Risk",
        y=factor
    )

    plt.title(f"{factor} vs Heart Attack Risk")
    plt.xlabel("Heart Attack Risk")
    plt.ylabel(factor)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 12. Correlation Analysis
# -----------------------------

correlation_columns = [
    "Age",
    "Diet_Score",
    "Cholesterol_Level",
    "Triglyceride_Level",
    "LDL_Level",
    "HDL_Level",
    "Systolic_BP",
    "Diastolic_BP",
    "Stress_Level",
    "Emergency_Response_Time",
    "Annual_Income"
]

correlation = df[correlation_columns].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Factors")

plt.tight_layout()
plt.show()


# -----------------------------
# 13. Select Features
# -----------------------------

features = [
    "Age",
    "Gender",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Smoking",
    "Alcohol_Consumption",
    "Physical_Activity",
    "Diet_Score",
    "Cholesterol_Level",
    "Triglyceride_Level",
    "LDL_Level",
    "HDL_Level",
    "Systolic_BP",
    "Diastolic_BP",
    "Air_Pollution_Exposure",
    "Family_History",
    "Stress_Level",
    "Healthcare_Access",
    "Heart_Attack_History",
    "Emergency_Response_Time",
    "Annual_Income",
    "Health_Insurance"
]

X = df[features]

y = df["Heart_Attack_Risk"]

print("\n===== MACHINE LEARNING DATA =====")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# -----------------------------
# 14. Identify Data Types
# -----------------------------

categorical_features = (
    X.select_dtypes(include=["object"])
    .columns
    .tolist()
)

numerical_features = (
    X.select_dtypes(exclude=["object"])
    .columns
    .tolist()
)

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# -----------------------------
# 15. Encode Categorical Data
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

print("\nOriginal feature shape:", X.shape)
print("Encoded feature shape:", X_encoded.shape)


# -----------------------------
# 16. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN TEST SPLIT =====")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------
# 17. Train Logistic Regression
# -----------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

print("\nModel trained successfully!")


# -----------------------------
# 18. Make Predictions
# -----------------------------

y_pred = model.predict(X_test)

print("\nPredictions completed successfully!")


# -----------------------------
# 19. Model Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===== MODEL PERFORMANCE =====")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------
# 20. Confusion Matrix
# -----------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Heart Attack Risk Prediction"
)

plt.tight_layout()
plt.show()


# -----------------------------
# 21. Feature Importance
# -----------------------------

feature_names = (
    preprocessor
    .get_feature_names_out()
)

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")

print(
    feature_importance
    .head(15)
)


# -----------------------------
# 22. Feature Importance Graph
# -----------------------------

top_features = (
    feature_importance
    .head(15)
    .sort_values("Coefficient")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.title(
    "Top 15 Factors Influencing Heart Attack Risk Prediction"
)

plt.xlabel("Model Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# -----------------------------
# 23. Save Model
# -----------------------------

joblib.dump(
    model,
    "../heart_risk_model.pkl"
)

joblib.dump(
    preprocessor,
    "../heart_risk_preprocessor.pkl"
)

print("\nModel saved successfully!")


# -----------------------------
# 24. Final Project Summary
# -----------------------------

print("\n========================================")
print("   HEART RISK ANALYTICS PROJECT")
print("========================================")

print(f"\nTotal Patients: {len(df)}")
print(f"Total Features: {len(features)}")

print("\nAlgorithm:")
print("Logistic Regression")

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nProject Status:")
print("EDA Completed")
print("Data Preparation Completed")
print("Model Training Completed")
print("Model Evaluation Completed")
print("Feature Analysis Completed")
print("Model Saved Successfully")

print("\n========================================")
print("Project execution completed!")
print("========================================")
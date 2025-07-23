# heart_disease_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Dataset Check (Confirm it's loaded properly)
# ------------------------------
try:
    df = pd.read_csv("heart.csv")
    print("✅ Dataset loaded successfully!")
    print("Shape of dataset:", df.shape)
    print("First 5 rows:\n", df.head())
except FileNotFoundError:
    print("❌ ERROR: 'heart.csv' not found in the current directory.")
    exit(1)

# ------------------------------
# Feature selection
# ------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("\n🔍 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print("5-Fold Cross-Validation Accuracy:", np.mean(cv_scores))

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=df.columns[:-1], class_names=["No Disease", "Disease"])
plt.title("Decision Tree for Heart Disease Prediction")
plt.show()

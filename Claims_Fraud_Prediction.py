# -*- coding: utf-8 -*-
"""
Updated on Wed Jan 24 2025
@author: Subhashri Ravichandran
"""

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Set plot style
plt.style.use('ggplot')

# Load Dataset
df = pd.read_csv("C:\\Users\\subha\\OneDrive\\Dokumente\\BCM\\insurance_claims.csv")

# Data Cleaning
df.replace('?', np.nan, inplace=True)
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])

# Drop unnecessary columns
drop_columns = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location', 'incident_date',
                'incident_state', 'incident_city', 'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', 
                'age', 'total_claim_amount', '_c39']
df.drop(drop_columns, inplace=True, axis=1)

# Define features and target variable
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'N' -> 0, 'Y' -> 1

# Encode categorical features in X
cat_cols = X.select_dtypes(include=['object']).columns
X[cat_cols] = X[cat_cols].apply(lambda col: label_encoder.fit_transform(col))

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=42)

# Standardize numeric columns
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_train.columns

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 140, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Train best model
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Feature Importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance_df)

# Evaluate Model Performance
print("\nClassification Report (Default Threshold):\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', color='blue', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Random Forest")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Optimal Threshold Selection
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.2f}")

# Adjust predictions using optimal threshold
y_pred_adjusted = (y_pred_proba >= optimal_threshold).astype(int)
print("\nClassification Report (Adjusted Threshold):\n")
print(classification_report(y_test, y_pred_adjusted))
# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance_df['Importance'], 
    y=feature_importance_df['Feature'], 
    palette='viridis'
)
plt.title('Feature Importance - Random Forest', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


























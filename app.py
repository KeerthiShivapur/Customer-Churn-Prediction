# train_and_predict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

# Step 1: Load dataset
df = pd.read_csv('customer_churn_dataset.csv')

# Step 2: Features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Step 3: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=15)

# Step 5: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression with GridSearchCV
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['liblinear', 'saga']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=cv)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# Step 7: Evaluate
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 8: Save model and scaler
pickle.dump(best_model, open("logistic_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Step 9: Manual user input from CLI (for testing backend)
print("\n🔢 Enter customer details to predict churn:")
tenure_months = int(input("Enter tenure in months (e.g., 12): "))
monthly_usage_hours = float(input("Enter monthly usage in hours (e.g., 28.5): "))
has_multiple_devices = int(input("Has multiple devices? (1 = Yes, 0 = No): "))
customer_support_calls = int(input("Number of customer support calls (e.g., 2): "))
payment_failures = int(input("Number of payment failures (e.g., 0): "))
is_premium_plan = int(input("Is premium plan? (1 = Yes, 0 = No): "))

# Step 10: Create input DataFrame
input_df = pd.DataFrame([{
    "tenure_months": tenure_months,
    "monthly_usage_hours": monthly_usage_hours,
    "has_multiple_devices": has_multiple_devices,
    "customer_support_calls": customer_support_calls,
    "payment_failures": payment_failures,
    "is_premium_plan": is_premium_plan
}])

# Step 11: Scale and Predict
input_scaled = scaler.transform(input_df)
prediction = best_model.predict(input_scaled)
print("\n🔍 Prediction:", "🔴 Churn" if prediction[0] == 1 else "🟢 Not Churn")

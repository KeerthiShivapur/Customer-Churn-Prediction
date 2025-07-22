import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv('customer_churn_dataset.csv')
df.head()
df['churn'].value_counts()
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X.shape
from imblearn.over_sampling import SMOTE
oversample=SMOTE()
X,y=oversample.fit_resample(X,y)
df['churn'].value_counts()
X.shape
len(y[y==0])
len(y[y==1])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
X_train
y_train
logistic=LogisticRegression(class_weight='balanced')
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
penalty=['l1','l2','elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg','lbfgs','liblinear','sag','saga']
p=dict(penalty=penalty,C=c_values,solver=solver)
from sklearn.model_selection import GridSearchCV,StratifiedKFold
grid=GridSearchCV(estimator=logistic,param_grid=p,scoring='accuracy')
grid.fit(X_train,y_train)
y_pred1=grid.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler=StandardScaler()
X_train1=scaler.fit_transform(X_train)
logisticreg=LogisticRegression(class_weight='balanced')
logisticreg.fit(X_train1,y_train)
y_pred2=logisticreg.predict(X_test1)
y_pred2=logisticreg.predict(X_test1)

print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
penalty=['l1','l2','elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg','lbfgs','liblinear','sag','saga']
p=dict(penalty=penalty,C=c_values,solver=solver)
from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold()
grid1=GridSearchCV(estimator=logisticreg,param_grid=p,scoring='accuracy',cv=cv)
grid1.fit(X_train1,y_train)
y_predict=grid1.predict(X_test1)
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
from sklearn.metrics import roc_auc_score
print("ROC AUC Score:", roc_auc_score(y_test, grid.predict_proba(X_test1)[:, 1]))
print(roc_auc_score(y_test,y_predict))
import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(grid,open('grid1.pkl','wb'))
pickle.dump(logisticreg,open('logisticreg.pkl','wb'))
### import pandas as pd

# Get input from the user
tenure_months = int(input("Enter tenure in months (e.g., 12): "))
monthly_usage_hours = float(input("Enter monthly usage in hours (e.g., 28.5): "))
has_multiple_devices = int(input("Has multiple devices? (1 = Yes, 0 = No): "))
customer_support_calls = int(input("Number of customer support calls (e.g., 2): "))
payment_failures = int(input("Number of payment failures (e.g., 0): "))
is_premium_plan = int(input("Is premium plan? (1 = Yes, 0 = No): "))

# Create DataFrame from inputs
input_data = pd.DataFrame([{
    "tenure_months": tenure_months,
    "monthly_usage_hours": monthly_usage_hours,
    "has_multiple_devices": has_multiple_devices,
    "customer_support_calls": customer_support_calls,
    "payment_failures": payment_failures,
    "is_premium_plan": is_premium_plan
}])

print("\n✅ User Input Data:")
print(input_data)
import joblib

# Load model
model = joblib.load("logisticreg.pkl")

# Predict
prediction = model.predict(input_df)
print("Prediction:", "Churn" if prediction[0] == 1 else "Not Churn")
import joblib

# Load model
model = joblib.load("logisticreg.pkl")

# Predict
prediction = model.predict(input_df)
print("Prediction:", "Churn" if prediction[0] == 1 else "Not Churn")

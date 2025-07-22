Customer Churn Prediction

This project predicts whether a customer is likely to churn (i.e., stop using a service) based on key customer behavior and account features such as tenure, usage, support interactions, and subscription type.

The model uses a trained machine learning algorithm to analyze these inputs and return a churn prediction.


Features Used

- `tenure_months` — How long the customer has been using the service (in months)
- `monthly_usage_hours` — Average usage hours per month
- `has_multiple_devices` — Whether the customer uses multiple devices (1 = Yes, 0 = No)
- `customer_support_calls` — Number of support calls in the recent period
- `payment_failures` — Count of failed payment attempts
- `is_premium_plan` — Whether the customer is on a premium subscription plan (1 = Yes, 0 = No)

---
Files Included

| File                     | Description                                 |
|--------------------------|---------------------------------------------|
| `churn_prediction.ipynb` | Jupyter Notebook that takes manual input and predicts churn |
| `churn_model.pkl`        | Trained machine learning model (Random Forest or similar) |
| `customer_churn.csv`     | Sample dataset used for training (optional) |
| `requirements.txt`       | List of required Python packages            |
| `README.md`              | Project documentation (this file)          |



How to Use

Requirements

Make sure you have Python installed with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

You can install them with:
```bash
pip install -r requirements.txt

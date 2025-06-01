import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Updated_Loan_Data.csv")

# Separate features and target
X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Sample 100 rows from training set for SHAP to be fast
sample_data = X_train.sample(100, random_state=42)

# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(sample_data)

# Plot SHAP summary plot
print("SHAP Summary Plot for XGBoost Model:")
shap.summary_plot(shap_values, sample_data)
plt.show()



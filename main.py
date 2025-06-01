print("Hello, Python is working!")
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

# Load dataset
df = pd.read_csv('Updated_Loan_Data.csv')

# Features and target
X = df.drop('Loan Status', axis=1)
y = df['Loan Status']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to hold models and results
models = {
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42)
}

# Function to evaluate a model
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n{name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

# Evaluate each model
for name, model in models.items():
    evaluate_model(name, model)
    df = pd.read_csv("Updated_Loan_Data.csv")
X = df.drop('Loan Status', axis=1)
y = df['Loan Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(random_state=42)
et_model = ExtraTreesClassifier(random_state=42)

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
et_model.fit(X_train, y_train)

# Sample data (small subset for SHAP speed)
sample_data = X_train.sample(100, random_state=42)

# SHAP TreeExplainers (CPU-friendly)
explainer_xgb = shap.TreeExplainer(xgb_model)
explainer_rf = shap.TreeExplainer(rf_model)
explainer_et = shap.TreeExplainer(et_model)

# SHAP values
shap_values_xgb = explainer_xgb.shap_values(sample_data)
shap_values_rf = explainer_rf.shap_values(sample_data)
shap_values_et = explainer_et.shap_values(sample_data)

# Plot summary (optional)
shap.summary_plot(shap_values_xgb, sample_data, show=False)
shap.summary_plot(shap_values_rf, sample_data, show=False)
shap.summary_plot(shap_values_et, sample_data, show=False)
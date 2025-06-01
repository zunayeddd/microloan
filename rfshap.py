import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("Updated_Loan_Data.csv")

# Separate features and target
X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sample some training data to make SHAP analysis faster
sample_data = X_train.sample(100, random_state=42)

# Create SHAP TreeExplainer for the trained Random Forest model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the sample data
shap_values = explainer.shap_values(sample_data)

# Generate SHAP summary plot
print("SHAP Summary Plot for Random Forest Model:")
shap.summary_plot(shap_values, sample_data)
plt.show()

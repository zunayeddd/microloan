import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import shap
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("Updated_Loan_Data.csv")

# Separate features and target
X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Extra Trees model
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)

# Take a sample from the training data for SHAP analysis (faster)
sample_data = X_train.sample(100, random_state=42)

# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(sample_data)

# Generate SHAP summary plot
print("SHAP Summary Plot for Extra Trees Model:")
shap.summary_plot(shap_values, sample_data)
plt.show()

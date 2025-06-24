# MicroLoan Default Prediction Using ML-Based Classification Models


This project focuses on building a machine learning-based system to predict loan default risks in microfinance using real-world financial data from a Bangladeshi bank. Our goal is to support microfinance institutions in making faster, more accurate, and less biased loan approval decisions.
📌 Project Objectives
Predict whether a micro loan applicant is likely to default.


Compare and evaluate multiple classification algorithms.
Identify the best model using key metrics: accuracy, precision, recall, F1 score.
Provide explainability using SHAP analysis.


Develop a foundation for future deployment in real-time applications.
🛠️ Algorithms Used
✅ XGBoost (Best-performing model)
🌲 Random Forest
🌳 Extra Trees


📊 Dataset
Real, anonymized borrower data from a Bangladeshi financial institution
Features include: monthly income, loan amount, credit history (CIB clearance), dependents, etc.
Imbalanced dataset: ~20% loan approvals, ~80% rejections


🔍 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
SHAP (for model interpretation)


🚀 Key Results
XGBoost achieved 99.30% accuracy, with high precision and recall
SHAP analysis revealed top features: Loan Amount, Gross Income, CIB Clearance


🧩 Future Work
Build a web application that allows microfinance staff to input borrower data and instantly get a risk prediction with SHAP-based explainability
Extend the model to other financial sectors and geographies


👥 Authors
Ahmed Md Zunayed
Bhuian Md Waliulla

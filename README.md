# House_Price_prediction
House Price Prediction using Machine Learning — explores Linear Regression, Ridge, Lasso, Decision Trees, Random Forests, Gradient Boosting &amp; XGBoost with feature engineering, and cross-validation for accurate house price estimation.

This project explores predictive modeling for house prices using multiple machine learning techniques. The dataset includes features such as bedrooms, bathrooms, square footage, year built, renovations, and location information (city, zipcode, etc.).

The goal is to build, evaluate, and compare models that can accurately estimate house prices, while also experimenting with feature engineering, feature selection, and advanced ensemble methods.

🚀 Project Workflow
1. Data Preprocessing

Handling missing values

Removing duplicates

Outlier detection & treatment (IQR, domain rules)

Encoding categorical features (city, statezip, street)

Feature scaling for linear models (StandardScaler)

2. Feature Engineering

house_age = Current year − yr_built

renovated_age = Current year − yr_renovated

has_basement = binary flag from sqft_basement

Quality score from waterfront, view, condition

Target encoding for high-cardinality categorical columns (city)

3. Feature Selection

Correlation-threshold filtering → drops highly correlated features

Recursive Feature Elimination (RFE, RFECV) → model-based feature ranking

Comparison of “all features” vs “selected features”

4. Models Implemented

Linear Models

Linear Regression

Ridge Regression

Lasso Regression

Tree-based Models

Decision Tree Regressor (with pruning & tuning)

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

5. Model Evaluation

Cross-validation (5-fold, train-only) for unbiased selection

Metrics used:

R² (explained variance)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Train vs Test comparison to check overfitting

📊 Experiments & Insights

Linear vs Regularized models → Ridge handled multicollinearity better, Lasso helped feature selection.

Decision Tree → Overfit heavily; pruning improved but still weaker.

Random Forest & Gradient Boosting → Much stronger performance, reduced variance.

XGBoost → Comparable to Gradient Boosting; faster & more regularized.

Feature Selection → RFE + correlation filtering improved model interpretability without much loss in accuracy.

⚙️ Tech Stack

Python 3.10+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

📈 Future Work

Hyperparameter tuning with RandomizedSearchCV / Optuna

Add SHAP/Permutation importance for feature explainability

Deploy trained model using Streamlit or Flask API

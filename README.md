# 🏡 House Price Prediction with Machine Learning

This project explores **predictive modeling for house prices** using multiple machine learning techniques.  
The dataset includes features such as bedrooms, bathrooms, square footage, year built, renovations, and location information (city, zipcode, etc.).  

The goal is to build, evaluate, and compare models that can accurately estimate house prices, while also experimenting with **feature engineering, feature selection, and advanced ensemble methods**.

---

## 🚀 Project Workflow

### 1. Data Preprocessing
- Handle missing values & duplicates  
- Outlier detection and treatment (IQR, domain rules)  
- Encode categorical features (`city`, `statezip`, `street`)  
- Feature scaling for linear models  

### 2. Feature Engineering
- `house_age` = Current year − `yr_built`  
- `renovated_age` = Current year − `yr_renovated`  
- `has_basement` = binary flag from `sqft_basement`  
- Quality score from `waterfront`, `view`, `condition`  
- Target encoding for high-cardinality categorical columns (`city`)  

### 3. Models Implemented
- **Linear Models**
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
- **Tree-based Models**
  - Decision Tree Regressor (with pruning & tuning)  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - XGBoost Regressor  

### 4. Model Evaluation
- **Cross-validation** (5-fold, train-only) for unbiased selection  
- Metrics used:  
  - **R²** (explained variance)  
  - **RMSE** (Root Mean Squared Error)  
  - **MAE** (Mean Absolute Error)  
- Train vs Test comparison to check overfitting  

---

## 📊 Experiments & Insights
- Ridge handled **multicollinearity** better, Lasso helped **feature selection**.  
- Decision Tree overfit heavily; pruning improved but still weaker.  
- Random Forest & Gradient Boosting gave stronger results.  
- XGBoost performed comparably to Gradient Boosting, with faster training and better regularization.  
- Feature selection (Correlation + RFE) improved interpretability without large accuracy loss.  

---

## ⚙️ Tech Stack
- **Python 3.10+**  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost  

---

## 📈 Future Work
- Hyperparameter tuning with `RandomizedSearchCV` or `Optuna`  
- Feature importance & SHAP for interpretability  
- Model deployment using **Streamlit** or **Flask API**  

---

## 🧑‍💻 How to Run
```bash
git clone <repo-url>
cd house-price-prediction
pip install -r requirements.txt

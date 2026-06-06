# Task-2-House-Price-Prediction
# Enhanced House Price Prediction System 🏠📊

An end-to-end Machine Learning pipeline utilizing the **California Housing Dataset** to predict median house values. This repository focuses on critical engineering workflows including feature scaling, training multiple regressor algorithms, and performing objective model evaluation to select the optimal model.

This project was developed as part of **Task 2: Feature Engineering, Model Optimization & Performance Comparison** at Maincrafts Technology.

---

## 🎯 Objectives
* **Data Preparation:** Correctly transform and handle data for Machine Learning ingestion.
* **Feature Scaling:** Implement robust preprocessing to ensure fair learning across features with vastly different numeric ranges.
* **Algorithm Exploration:** Train and tune multiple regression models beyond a basic baseline.
* **Performance Comparison:** Professionally evaluate models using key metrics ($RMSE$ and $R^2$ Score) to justify the best-performing selection.

---

## 🛠️ Tools & Technologies Used
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Data Manipulation:** pandas, NumPy
* **Machine Learning:** scikit-learn
* **Data Visualization:** matplotlib, seaborn

---

## 📋 Machine Learning Workflow

### 1. Preprocessing & Feature Scaling
Real-world features like `Population` and `House Age` exist on entirely different ranges. To prevent high-magnitude features from dominating the learning phase and making gradients unstable, a `StandardScaler` was applied to standardize the features to a common scale (mean = 0, variance = 1).

### 2. Train-Test Split
The scaled dataset was split into an **80% training set** and a **20% testing set** using a fixed `random_state=42` to guarantee reproducibility and prevent data leakage.

### 3. Evaluated Models
Three diverse regression algorithms were trained and compared:
* **Linear Regression:** Implemented to establish a baseline performance metric.
* **Ridge Regression:** Introduced to apply L2 regularization and prevent potential overfitting.
* **Decision Tree Regressor:** Configured with a `max_depth=5` to successfully capture non-linear relationships within the features.

---

## 📈 Evaluation & Results

Models were strictly cross-evaluated using **Root Mean Squared Error (RMSE)** for absolute error magnitude tracking, and the **Coefficient of Determination ($R^2$ Score)** to measure explanatory variance power.

### Model Performance Summary

| Algorithm | RMSE (Lower is Better) | $R^2$ Score (Higher is Better) |
| :--- | :---: | :---: |
| **Linear Regression** | ~0.745 | ~0.575 |
| **Ridge Regression** | ~0.746 | ~0.575 |
| **Decision Tree Regressor** | ~0.594 | ~0.729 |

### Visual Validation
A scatter plot of **Actual vs. Predicted House Prices** was generated against a perfect reference diagonal line ($y = x$) to visually audit the alignment and error dispersion of the predictions.

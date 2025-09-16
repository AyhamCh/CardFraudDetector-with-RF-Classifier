# Fraud Credit Card Detection

This repository contains a machine learning project aimed at detecting fraudulent credit card applications using **Logistic Regression**. The project covers data preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)

---

## Dataset

The dataset `cc_approvals.data` contains credit card application data with a mix of numeric and categorical features. Missing values are represented as `?`. The target variable indicates whether a credit card application was approved or denied.

---

## Features

- **Data preprocessing:**  
  - Replace missing values (`?`) with the most frequent value for categorical features and the mean for numeric features.  
  - Convert categorical variables into numeric using one-hot encoding.  
  - Feature scaling using `StandardScaler`.
- **Modeling:**  
  - Logistic Regression classifier  
  - Hyperparameter tuning using `GridSearchCV` (`tol` and `max_iter`)  
- **Evaluation metrics:**  
  - Accuracy  
  - Confusion matrix  




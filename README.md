# Airbnb Price Category Prediction Using Machine Learning

## Overview

This project develops a machine learning pipeline to classify Airbnb listings into high- and low-price categories using property characteristics, host information, and review metrics. The goal is to identify key factors associated with Airbnb pricing and build predictive models that can support hospitality investment and market analysis decisions.

The project explores the full machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, model development, hyperparameter tuning, and performance evaluation. Multiple classification approaches were compared, including traditional machine learning models and a neural network.

---

## Problem Statement

Airbnb listing prices are influenced by a variety of factors, including property features, location, host characteristics, and customer reviews. Understanding these relationships can help hospitality analysts identify valuable listings and better understand pricing patterns.

This project addresses the following question:

**Can machine learning models accurately classify Airbnb listings into pricing categories based on available listing information?**

The task was formulated as a binary classification problem:

- **High-priced listing:** Listing price above the selected threshold
- **Low-priced listing:** Listing price below the selected threshold

---

## Dataset

The project uses the **New York City Airbnb Open Data** dataset, which contains information about Airbnb listings, including:

- Listing details
- Room type and property characteristics
- Host information
- Availability data
- Review scores
- Pricing information

The original price variable was processed to reduce the impact of extreme outliers and transformed into a binary classification target.

---

## Data Preprocessing & Feature Engineering

Real-world datasets often contain missing values, inconsistent formats, and high-dimensional categorical features. The following preprocessing steps were performed:

- Removed unnecessary text-heavy columns with limited predictive value
- Handled missing values using appropriate imputation techniques
- Converted categorical variables into numerical representations using encoding methods
- Converted boolean variables into numerical features
- Scaled numerical features for neural network training
- Applied winsorization to reduce the effect of extreme price values
- Used feature selection techniques to identify important predictive features

The final dataset was prepared for both traditional machine learning models and neural network training.

---

## Exploratory Data Analysis

Exploratory analysis was performed to understand patterns within the dataset, including:

- Distribution of listing prices
- Class balance between price categories
- Relationships between numerical features
- Feature correlations
- Differences between high- and low-priced listings

Key observations included the presence of significant price outliers and relationships between listing characteristics, reviews, and pricing categories.

---

## Machine Learning Models

Three classification approaches were evaluated:

### Logistic Regression
Used as a baseline model due to its interpretability and ability to establish a performance benchmark.

### Decision Tree Classifier
Used to capture nonlinear relationships while maintaining interpretability through feature importance analysis.

### Neural Network
A feed-forward neural network was implemented using TensorFlow/Keras to learn more complex relationships within the feature space.

Neural Network Architecture:

```
Input Layer
      ↓
Dense Layer (64 units, ReLU)
      ↓
Dense Layer (32 units, ReLU)
      ↓
Dense Layer (16 units, ReLU)
      ↓
Output Layer (Sigmoid)
```

---

## Model Evaluation

Models were evaluated using:

- Accuracy
- F1 Score

F1 score was emphasized because it provides a better measure of performance when considering both false positives and false negatives.

### Results

| Model | Accuracy | F1 Score |
|------|----------|----------|
| Decision Tree | 82.5% | 59.5% |
| Neural Network | 84.9% | 67.5% |

The neural network achieved the strongest overall performance, improving both accuracy and F1 score compared to the decision tree. The higher F1 score indicates that the neural network was better able to identify high-priced listings while balancing classification errors.

---

## Key Findings

- Machine learning models can effectively identify pricing categories from Airbnb listing characteristics.
- Feature preprocessing and selection significantly impact model performance.
- Neural networks achieved higher predictive performance, while decision trees provided greater interpretability.
- The results demonstrate the tradeoff between model complexity and explainability.

---

## Technologies Used

### Programming Language
- Python

### Data Analysis & Visualization
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
  - Logistic Regression
  - Decision Trees
  - Feature Selection
  - GridSearchCV
  - Model Evaluation

### Deep Learning
- TensorFlow
- Keras

### Development Environment
- Jupyter Notebook

---

## Future Improvements

Potential improvements include:

- Deploying the model as an interactive web application using Streamlit
- Adding SHAP analysis for improved model interpretability
- Incorporating geographic features such as neighborhood-level information
- Testing ensemble methods such as Random Forest and XGBoost
- Expanding the model into a pricing recommendation system

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for solving a real-world classification problem. By comparing interpretable machine learning models with neural networks, the project highlights the importance of balancing predictive performance, model complexity, and explainability when developing machine learning solutions.

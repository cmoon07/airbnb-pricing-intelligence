# Airbnb Price Category Prediction Using Machine Learning

## Overview

This project develops a machine learning pipeline to classify NYC Airbnb listings into high-priced (top 25%) and low-price categories using property characteristics, host information, and review metrics. The goal is to identify key factors associated with Airbnb pricing and build predictive models that can support hospitality investment and market analysis decisions.

The project explores the full machine learning workflow, including exploratory data analysis, cleaning, encoding, model training, hyperparameter tuning, and performance evaluation. Multiple classification approaches were compared, including Logistic Regression, a Decision Tree, and a feed-forward neural network against a majority-class baseline.

---

## Problem Statement

A hospitality investment group sorts short-term rentals into pricing tiers by hand using neighborhood reputation, room type, and host reviews. That process is slow and inconsistent as the portfolio grows.

This project addresses the following question:

**Can machine learning models accurately classify Airbnb listings into pricing categories based on available listing information?**

The task was formulated as a binary classification problem:

- **High-priced listing:** Listing price above the 75th percentile ($180/night)
- **Low-priced listing:** Listing price below the 75th percentile

---

## Dataset

The project uses the **New York City Airbnb Open Data** dataset. The data is included as data_capstone/airbnbListingsData.zip and the notebook reads it directly. The dataset contains information about Airbnb listings, including:

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

- Dropped long free-text columns (description, neighborhood_overview, host_about) that were mostly missing or would need NLP
- Filled missing host_response_rate, host_acceptance_rate, bedrooms, and beds with column means
- One-hot encoded neighbourhood_group_cleansed, room_type, instant_bookable, and has_availability
- Kept the 48 numeric and encoded columns as model inputs, excluding price
- Standardized features with StandardScaler (fit on training data only) for the neural network
- Split 67% train / 33% test with a fixed random seed

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
Interpretable model, with features standardized inside a pipeline.

### Decision Tree Classifier
Tuned with 5-fold grid search over max_depth and min_samples_leaf together, optimizing F1 on the training split.

### Neural Network
Three hidden layers with ReLU and a sigmoid output, trained with SGD (learning rate 0.001) for 200 epochs with a 20% validation split.

Neural Network Architecture:

```
Input Layer (48 features)
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
- ROC-AUC

F1 score was emphasized because it provides a better measure of performance when considering both false positives and false negatives. Especially since it's only about a quarter of listings that are high-priced, and a model can look accurate while missing most of them. 

### Results

| Model | Accuracy | F1 Score | ROC-AUC |
|------|----------|----------|----------|
| Majority-class baseline (always "low) | 74.3% | 0.000 | 0.500 |
| Logistic Regression | 83.2% | 0.619 | 0.870 |
| Decision Tree | 82.8% | 0.634 | 0.872 |
| Neural Network | 84.7% | 0.668 | 0.893 |

All three models clearly beat the baseline and rank listings well (ROC-AUC 0.87-0.89). The neural network achieved the strongest overall performance, improving both accuracy and F1 score compared to the decision tree. However, its lead is modest with about 2 points of accuracy and 3-5 points of F1. Logistic Regression and Decision Tree come close while being far easier to explain, so they're strong choices when every prediction must be justified.

---

## Known Limitations

- Results come from a single train/test split, so differences of 1-2 points between models may be partly noise (cross-validation would confirm the ranking)
- Mean imputation ignores whether a value was missing, which can itself carry signal
- Borough and neighborhood may act as proxies for income and demographics, so errors should be audited by borough before any deployment

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

- Tune the decision threshold to the client's cost of missing a premium listing
- Confirm the model ranking with cross-validation
- Try gradient-boosted trees (for example, XGBoost or LightGBM)
- Audit error rates by borough before any deployment

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for solving a real-world classification problem. By comparing interpretable machine learning models with neural networks, the project highlights the importance of balancing predictive performance, model complexity, and explainability when developing machine learning solutions.

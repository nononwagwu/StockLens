# StockLens

StockLens is a machine learning project that analyzes company financial indicators to predict stock performance. The project was developed as part of an AI4ALL program to apply machine learning techniques to a real-world financial problem.

## Overview

Financial datasets contain hundreds of indicators that can potentially provide insight into company and stock performance. StockLens explores how machine learning models can identify patterns across these financial features and generate predictions from historical company data.

The project works with a dataset containing over **20,000 financial observations** and **200+ features** spanning multiple years.

## Machine Learning Pipeline

The project follows an end-to-end machine learning workflow:

- Cleaned and preprocessed large-scale financial data
- Handled missing values and filtered outliers
- Prepared financial features for model training
- Trained and compared multiple regression models
- Used 5-fold cross-validation to evaluate model performance
- Evaluated predictions using regression metrics such as Mean Absolute Error (MAE)
- Built an interactive Streamlit interface for exploring model results

## Models

Multiple regression approaches were evaluated, including:

- Linear Regression
- Random Forest Regressor
- Additional regression models explored during model development

The models were benchmarked using cross-validation to compare their ability to predict continuous stock performance outcomes.

## Tech Stack

- **Python**
- **pandas** — data processing and analysis
- **NumPy** — numerical computing
- **scikit-learn** — preprocessing, model training, and evaluation
- **Matplotlib** — data visualization
- **Streamlit** — interactive application and model visualization

## Dataset

The project uses historical financial data containing financial indicators from publicly traded U.S. companies between 2014 and 2018.

The dataset includes features related to:

- Revenue and profitability
- Assets and liabilities
- Cash flow
- Financial ratios
- Company growth metrics
- Stock performance

## My Contributions

- Developed preprocessing pipelines for 20,000+ observations and 200+ financial features
- Applied missing-value handling and outlier filtering to prepare data for modeling
- Trained and evaluated regression models using 5-fold cross-validation
- Compared model performance against a Linear Regression baseline
- Contributed to the development of the Streamlit interface for presenting model insights

## Future Improvements

- Experiment with additional ensemble and gradient boosting models
- Perform more extensive feature selection and feature engineering
- Expand model explainability using feature importance and SHAP
- Improve the Streamlit interface and interactive visualizations
- Evaluate performance on more recent financial datasets

## Team

Developed collaboratively as part of an AI4ALL machine learning program.

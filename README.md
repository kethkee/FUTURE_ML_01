# FUTURE_ML_01 – Sales & Demand Forecasting

## Project Overview
This project builds a Machine Learning system to forecast future sales based on historical retail data. The goal is to help businesses plan inventory, staffing, and financial decisions using data-driven insights.

## Dataset
Superstore Sales Dataset (2014–2017)

Features include:
- Order Date
- Sales
- Category
- Region
- Profit

## Steps Performed

### 1. Data Preparation
- Converted Order Date to datetime
- Aggregated daily sales
- Converted daily sales into monthly sales

### 2. Feature Engineering
- Extracted Month
- Created TimeIndex to capture trend

### 3. Model Training
Linear Regression was used as the baseline model to capture trend and seasonal effects.

### 4. Model Evaluation
Evaluation metrics used:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)

### 5. Forecasting
The model predicts **future sales for the next 6 months**.

### 6. Visualization
A dashboard was created using **Streamlit** to visualize:
- Historical sales
- Model predictions
- Future forecasts

## Business Insights
- Sales show an overall upward trend.
- Seasonal spikes occur toward year-end.
- Forecasting can help businesses plan inventory and staffing.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

# Inflation Prediction Using Hybrid Models: Streamlit Application

This repository contains a Streamlit application designed to predict inflation rates using a hybrid model that integrates traditional forecasting methods, machine learning algorithms, and deep learning techniques. The app consists of two pages:
1. **Exploratory Data Analysis (EDA)**: Provides interactive visualizations and insights into the underlying macroeconomic data.
2. **Hybrid Model Prediction**: Allows users to explore inflation predictions made by the hybrid model combining ARIMA, RF, SVR, and LSTM.

## Project Overview

The objective of this project is to develop a robust inflation prediction model by combining various forecasting techniques. The models used in this project include:
- **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variable) for traditional forecasting
- **Long Short-Term Memory (LSTM)** networks for deep learning

The hybrid approach leverages the strengths of each technique to improve prediction accuracy, incorporating external factors like housing market fluctuations and immigration trends.

## Pages in the Application

### 1. **Exploratory Data Analysis (EDA)**
   - **Data Visualizations**: Interactive plots of macroeconomic variables like inflation, immigration, housing prices, and stock market data.
   - **Statistical Insights**: Descriptive statistics and correlation analysis for the variables.
   - **Time Series Analysis**: Decomposition and trend analysis of inflation and other time series data.

### 2. **Hybrid Model Prediction**
   - **Hybrid Approach**: Combines SARIMAX and LSTM models to predict inflation rates.
   - **Model Evaluation**: Displays performance metrics such as MAE, MSE, RMSE, and MAPE for each model and the hybrid model.
   - **Prediction Output**: Visualizes the inflation predictions and compares them with the actual values.


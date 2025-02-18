# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:19:57 2025

@author: Subhashri ravichandran
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Read CSV
df = pd.read_csv("C:\\Users\\subha\\OneDrive\\Dokumente\\BCM\\WinterSemester\\Data Science\\Project\\Kaggle\\price_paid_records.csv")
pd.set_option('display.max_columns', None)

# Drop unwanted features
df.drop(columns=['Transaction unique identifier', 'Duration', 'PPDCategory Type', 'Record Status - monthly file only'], inplace=True)

# Convert 'Date of Transfer' to datetime object and extract year and month
df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'].apply(lambda x: x.split(' ')[0]), format='%Y-%m-%d')
df['Year'] = df['Date of Transfer'].dt.year
df['Month'] = df['Date of Transfer'].dt.month
df['YearMonth'] = df['Date of Transfer'].dt.to_period('M')

# Filter data for 1995-2015
df_filtered = df[(df['Year'] >= 1995) & (df['Year'] <= 2015)]

# Aggregate data monthly
monthly_data = df_filtered.groupby('YearMonth').agg({'Price': 'mean', 'Date of Transfer': 'count'}).reset_index()
monthly_data.rename(columns={'Date of Transfer': 'Transactions'}, inplace=True)

# Convert YearMonth to datetime
monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()

# Split into training (1995-2015)
train_data_monthly = monthly_data.set_index('YearMonth')
price_train_monthly = train_data_monthly['Price']
transaction_train_monthly = train_data_monthly['Transactions']

# Define a function to forecast monthly data using SARIMA
def forecast_sarima_monthly(data, forecast_periods, seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_periods)
    
    # Generate forecast index starting from the next month after the last month in the training set
    forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
    
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    return forecast_index, forecast_values, forecast_ci

# Cross-validation with TimeSeriesSplit
def cross_validate_sarima(data, n_splits=5, forecast_periods=12, seasonal_order=(1, 1, 1, 12)):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    
    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        
        # Forecast the data using SARIMA
        forecast_index, forecast_values, _ = forecast_sarima_monthly(train, forecast_periods, seasonal_order)
        
        # Calculate RMSE for each fold
        rmse = np.sqrt(((forecast_values - test[:forecast_periods]) ** 2).mean())
        rmses.append(rmse)
    
    return np.mean(rmses), rmses

# Perform cross-validation for Price and Transactions
price_rmse, price_rmses = cross_validate_sarima(price_train_monthly, n_splits=5, forecast_periods=12)
transaction_rmse, transaction_rmses = cross_validate_sarima(transaction_train_monthly, n_splits=5, forecast_periods=12)

# Output RMSE values for cross-validation
print(f"Average RMSE for Price: {price_rmse}")
print(f"RMSE for each fold (Price): {price_rmses}")
print(f"Average RMSE for Transactions: {transaction_rmse}")
print(f"RMSE for each fold (Transactions): {transaction_rmses}")

# Forecast for 2016-2022 (monthly)
forecast_months = 84  # Forecast from Jan 2016 to Dec 2022
price_forecast_index_monthly, price_forecast_values_monthly, price_forecast_ci_monthly = forecast_sarima_monthly(price_train_monthly, forecast_months)
transaction_forecast_index_monthly, transaction_forecast_values_monthly, transaction_forecast_ci_monthly = forecast_sarima_monthly(transaction_train_monthly, forecast_months)

# Combine results for monthly forecast
forecast_results_monthly = pd.DataFrame({
    'YearMonth': price_forecast_index_monthly,
    'Forecasted Price': price_forecast_values_monthly,
    'Forecasted Transactions': transaction_forecast_values_monthly
})

# Visualization of monthly forecasts for prices
plt.figure(figsize=(12, 6))
plt.plot(price_train_monthly, label='Historical Prices', color='blue')
plt.plot(price_forecast_index_monthly, price_forecast_values_monthly, label='Forecasted Prices', color='orange')
plt.fill_between(price_forecast_index_monthly, price_forecast_ci_monthly.iloc[:, 0], price_forecast_ci_monthly.iloc[:, 1], color='orange', alpha=0.2)
plt.title('Historical and Future Trend of Avg House Prices (1995â€“2022)')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.legend()
plt.show()

# Visualization of monthly forecasts for transactions
plt.figure(figsize=(12, 6))
plt.plot(transaction_train_monthly, label='Historical Transactions', color='green')
plt.plot(transaction_forecast_index_monthly, transaction_forecast_values_monthly, label='Forecasted Transactions', color='red')
plt.fill_between(transaction_forecast_index_monthly, transaction_forecast_ci_monthly.iloc[:, 0], transaction_forecast_ci_monthly.iloc[:, 1], color='red', alpha=0.2)
plt.title('Historical and Future Trend of Transactions Count (1995â€“2022)')
plt.xlabel('Year')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()

# Display forecasted monthly data
print(forecast_results_monthly)

# Create a copy of the filtered data to avoid the view vs copy warning
forecasted_periods = forecast_results_monthly[(forecast_results_monthly['YearMonth'].dt.year >= 2017) & (forecast_results_monthly['YearMonth'].dt.year <= 2022)].copy()

# Extract Year and Month from the 'YearMonth' column
forecasted_periods['Year'] = forecasted_periods['YearMonth'].dt.year
forecasted_periods['Month'] = forecasted_periods['YearMonth'].dt.month

# Ensure that 'Forecasted Transactions' and 'Forecasted Price' are available in forecasted_periods
if 'Forecasted Transactions' in forecasted_periods.columns and 'Forecasted Price' in forecasted_periods.columns:
# Pivot tables for transactions and prices (swapped Month and Year)
    transactions_pivot = forecasted_periods.pivot(index='Year', columns='Month', values='Forecasted Transactions')
    prices_pivot = forecasted_periods.pivot(index='Year', columns='Month', values='Forecasted Price')

  # Plot Heatmap for Forecasted Transactions (Year on y-axis, Month on x-axis)
    plt.figure(figsize=(12, 6))
    sns.heatmap(transactions_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
    plt.title('Forecasted Monthly Transactions (2017â€“2022)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

    # Plot Heatmap for Forecasted Prices (Year on y-axis, Month on x-axis) with "coolwarm" color palette
    plt.figure(figsize=(12, 6))
    sns.heatmap(prices_pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5)
    plt.title('Forecasted Monthly Average Prices (2017â€“2022)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Columns 'Forecasted Transactions' or 'Forecasted Price' not found in the DataFrame.")


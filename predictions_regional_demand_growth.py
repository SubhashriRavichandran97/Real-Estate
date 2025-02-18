# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:27:28 2025

@author: subhashri ravichandran
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Group by Town/City and YearMonth to calculate monthly transaction counts
city_monthly_transactions = df_filtered.groupby(['Town/City', 'YearMonth'])['Date of Transfer'].count().reset_index()
city_monthly_transactions.rename(columns={'Date of Transfer': 'Transactions'}, inplace=True)

# Convert YearMonth to datetime
city_monthly_transactions['YearMonth'] = city_monthly_transactions['YearMonth'].dt.to_timestamp()

# Define a function to forecast transactions for each city
def forecast_city_transactions(data, city, forecast_periods, seasonal_order=(1, 1, 1, 12)):
    city_data = data[data['Town/City'] == city].set_index('YearMonth')
    transaction_series = city_data['Transactions']

    model = SARIMAX(transaction_series, order=(1, 1, 1), seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_periods)

    forecast_index = pd.date_range(start=transaction_series.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    return forecast_index, forecast_values, forecast_ci, results

# Forecast for top 15 cities with the most transactions
forecast_periods = 84  # 7 years (2016-2022)
top_cities = city_monthly_transactions.groupby('Town/City')['Transactions'].sum().nlargest(15).index

forecast_results = []
validation_errors = []
for city in top_cities:
    forecast_index, forecast_values, forecast_ci, results = forecast_city_transactions(city_monthly_transactions, city, forecast_periods)
    forecast_df = pd.DataFrame({
        'YearMonth': forecast_index,
        'City': city,
        'Forecasted Transactions': forecast_values
    })
    forecast_results.append(forecast_df)

    # Validation: Calculate errors using historical data
    city_data = city_monthly_transactions[city_monthly_transactions['Town/City'] == city].set_index('YearMonth')
    historical_data = city_data['Transactions']
    if len(historical_data) > 24:  # Ensure sufficient data for validation
        validation_data = historical_data[-24:]
        train_data = historical_data[:-24]

        # Refit model on training data
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        validation_results = model.fit(disp=False)
        validation_forecast = validation_results.get_forecast(steps=24)

        # Calculate errors
        validation_forecast_values = validation_forecast.predicted_mean
        mae = mean_absolute_error(validation_data, validation_forecast_values)
        rmse = np.sqrt(mean_squared_error(validation_data, validation_forecast_values))
        validation_errors.append({'City': city, 'MAE': mae, 'RMSE': rmse})

forecast_results = pd.concat(forecast_results)
validation_errors = pd.DataFrame(validation_errors)
print(validation_errors)

# Calculate Yearly Transactions (sum of monthly transactions)
forecast_results['Year'] = forecast_results['YearMonth'].dt.year
forecast_results['Month'] = forecast_results['YearMonth'].dt.month
forecast_results['Yearly Transactions'] = forecast_results.groupby(['City', 'Year'])['Forecasted Transactions'].transform('sum')

# Calculate Year-over-Year Growth for each city
forecast_results['Previous Year Transactions'] = forecast_results.groupby('City')['Yearly Transactions'].shift(1)
forecast_results['YoY Growth (%)'] = ((forecast_results['Yearly Transactions'] - forecast_results['Previous Year Transactions']) / forecast_results['Previous Year Transactions']) * 100

# Filter forecast_results for years 2018-2022
forecast_results_filtered = forecast_results[(forecast_results['Year'] >= 2018) & (forecast_results['Year'] <= 2022)]

# Bar Chart for Top 15 Cities by Average YoY Growth (2018-2022)
avg_growth = forecast_results_filtered.groupby('City')['YoY Growth (%)'].mean().reset_index()
avg_growth = avg_growth.nlargest(15, 'YoY Growth (%)')

plt.figure(figsize=(12, 6))
plt.bar(avg_growth['City'], avg_growth['YoY Growth (%)'], color='skyblue')
plt.title('Top 15 Cities by Average YoY Growth (2018â€“2022)', fontsize=16)
plt.xlabel('City', fontsize=12)
plt.ylabel('Average YoY Growth (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



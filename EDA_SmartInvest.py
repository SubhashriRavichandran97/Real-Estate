# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:16:36 2024
---
@author: subhashri ravichandran
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker
import numpy as np
from pandarallel import pandarallel

# property type mapping
property_type_mapping = {
    'D': 'Detached',
    'S': 'Semi-Detached',
    'T': 'Terraced',
    'F': 'Flats',
    'O': 'Other'
}

# read csv
df = pd.read_csv("C:\\Users\\subha\\OneDrive\\Dokumente\\BCM\\WinterSemester\\Data Science\\Project\\Kaggle\\price_paid_records.csv")
pd.set_option('display.max_columns', None)
df.head()
df.columns
df.describe()
df.info()
df.isnull().sum()
df.shape
df.columns

# drop unwanted features
df.drop(columns = 'Transaction unique identifier', axis = 1, inplace = True)
df.drop(columns = 'Duration', axis = 1, inplace = True)
df.drop(columns = 'PPDCategory Type', axis = 1, inplace = True)
df.drop(columns = 'Record Status - monthly file only', axis = 1, inplace = True)
df.shape
df.head()

# convert 'Date of Transfer' to datetime object 
pandarallel.initialize()
df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x : x.split(' ')[0])
df['Year'] = df['Date of Transfer'].str[:4]
df['Date of Transfer'].iloc[1].split(" ")[0]
df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x: __import__('datetime').datetime.strptime(x, '%Y-%m-%d'))
df['Year'] = df['Date of Transfer'].dt.year
df['Month'] = df['Date of Transfer'].dt.month
df.head()
print('No of cities:', df['Town/City'].nunique())

# overall price trend analysis (2007-2017)
df_filtered = df[(df['Year'] >= 2007) & (df['Year'] <= 2017)]
price_trend = df_filtered.groupby('Year')['Price'].mean()
plt.figure(figsize=(12, 6))
price_trend.plot(marker='o', color='orange')
plt.title('Average House Prices Over Time')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.show()

# overall transaction trend analysis(2007-2017)
df_filtered = df[df['Year'].between(2007, 2017)]
transaction_counts = df_filtered.groupby([df_filtered['Year'], df_filtered['Month']]).size().unstack(fill_value=0).astype(int)
plt.figure(figsize=(12, 6))
sns.heatmap(transaction_counts, cmap='YlGnBu', annot=True, fmt="d", linewidths=.5)
plt.title('Heatmap of Transactions Across Months and Years (2007-2017)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

# property analysis
# price trend for new vs old property through the years 2007 to 2017
df_filtered = df[(df['Year'] >= 2007) & (df['Year'] <= 2017)]
old_new_comparison = df_filtered.groupby(['Old/New', 'Year'])['Price'].mean().unstack(level=0)
old_new_comparison.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
plt.title('Average Price of Old vs New Properties by Year (2007-2017)')
plt.ylabel('Average Price')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend(title='Property Type', loc='upper left', labels=['Old Properties', 'New Properties'])
plt.tight_layout()
plt.show()

# property analysis
# transaction trend for old vs new property through the years 2007 to 2017
df_filtered = df[(df['Year'] >= 2007) & (df['Year'] <= 2017)]
old_new_transactions = df_filtered.groupby(['Old/New', 'Year']).size().unstack(level=0)
ax = old_new_transactions.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
plt.title('Number of Transactions for Old vs New Properties by Year (2007-2017)')
plt.ylabel('Number of Transactions')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend(title='Property Type', loc='upper right', labels=['Old Properties', 'New Properties'])
plt.tight_layout()
plt.show()

# property type analysis
# Price trends for different property types (2007-2017)
# Property type mapping

df_filtered = df[(df['Year'] >= 2007) & (df['Year'] <= 2017)]
price_trends = df_filtered.groupby(['Property Type', 'Year'])['Price'].mean().unstack(level=0)
price_trends.columns = price_trends.columns.map(property_type_mapping)
price_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Price Trends for Different Property Types (2007-2017)')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.legend(title='Property Type')
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()

# property type analysis
# transaction trends for different property types (2007-2017)
df_filtered = df[(df['Year'] >= 2007) & (df['Year'] <= 2017)]
transaction_counts = df_filtered.groupby(['Property Type', 'Year']).size().unstack(level=0)
transaction_counts.columns = transaction_counts.columns.map(property_type_mapping)
transaction_counts.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Number of Transactions for Each Property Type (2007-2017)')
plt.xlabel('Year')
plt.ylabel('Number of Transactions')
plt.legend(title='Property Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# seasonal analysis
# Comparison of price and transaction trend by month (2015-2017)
df_filtered = df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]
monthly_transaction_count = df_filtered.groupby(['Month', 'Year']).size().unstack(fill_value=0)
monthly_avg_price = df_filtered.groupby('Month')['Price'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, ax1 = plt.subplots(figsize=(12, 6))
width = 0.25
x = np.arange(1, 13)
years = monthly_transaction_count.columns
for i, year in enumerate(years):
    ax1.bar(x + i * width - width, monthly_transaction_count[year], width=width, label=f'{year}', alpha=0.7)
ax1.set_ylabel('Number of Transactions', color='orange', fontsize=12)
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_xlabel('Month')
ax1.set_xticks(x)
ax1.set_xticklabels(month_names, fontsize=10)
ax2 = ax1.twinx()
ax2.plot(x, monthly_avg_price, color='purple', marker='o', linewidth=2, label='Average Price')
ax2.set_ylabel('Average Price', color='purple', fontsize=12)
ax2.tick_params(axis='y', labelcolor='purple')
plt.title('Comparison of Average Price Trend and Number of Transactions by Month (2015-2017)', fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)
fig.tight_layout()
plt.show()

# regional analysis
# top 15 cities with most expensive and least expensive properties
city_prices = df.groupby('Town/City')['Price'].mean().sort_values()
least_expensive_cities = city_prices.head(15)
most_expensive_cities = city_prices.tail(15)
plt.figure(figsize=(10, 6))
least_expensive_cities.plot(kind='bar', color='blue')
plt.title('Top 15 least expensive cities for property prices')
plt.ylabel('Average Price')
plt.xlabel('City')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
most_expensive_cities.plot(kind='bar', color='green')
plt.title('Top 15 most expensive cities for property prices')
plt.ylabel('Average Price')
plt.xlabel('City')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

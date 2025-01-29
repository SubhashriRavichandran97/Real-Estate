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
import seaborn as sns
import numpy as np
from pandarallel import pandarallel

# read csv
df = pd.read_csv("C:\\Users\\subha\\OneDrive\\Dokumente\\BCM\\WinterSemester\\Data Science\\Project\\Kaggle\\price_paid_records.csv")
pd.set_option('display.max_columns', None)

# Function to filter data based on year range
def filter_data_by_year(df, start_year, end_year):
    return df[df['Year'].between(start_year, end_year)]

# map property code to names
property_type_mapping = {
     'D': 'Detached',
     'S': 'Semi-Detached',
     'T': 'Terraced',
     'F': 'Flats',
     'O': 'Other'
}

# Function to plot a heatmap for transaction counts
def plot_transaction_heatmap(df_filtered):
    transaction_counts = df_filtered.groupby([df_filtered['Year'], df_filtered['Month']]).size().unstack(fill_value=0).astype(int)
    plt.figure(figsize=(12, 6))
    sns.heatmap(transaction_counts, cmap='YlGnBu', annot=True, fmt="d", linewidths=.5)
    plt.title('Heatmap of Transactions Across Months and Years(2007-2017)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.show()

# Function to plot old vs new property price trends
def plot_old_new_property_price_trends(df_filtered):
    old_new_comparison = df_filtered.groupby(['Old/New', 'Year'])['Price'].mean().unstack(level=0)
    old_new_comparison.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
    plt.title('Average Price of Old vs New Properties by Year(2007-2017)')
    plt.ylabel('Average Price')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.legend(title='Property Type', loc='upper left', labels=['Old Properties', 'New Properties'])
    plt.tight_layout()
    plt.show()

# Function to plot transaction trends for old vs new properties
def plot_old_new_property_transaction_trends(df_filtered):
    transaction_counts = df_filtered.groupby(['Old/New', 'Year']).size().unstack(level=0)
    transaction_counts.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
    plt.title('Number of Transactions for Old vs New Properties by Year (2007-2017)')
    plt.xlabel('Year')
    plt.ylabel('Number of Transactions')
    plt.legend(title='Property Type', labels=['Old Properties', 'New Properties'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot price trends for different property types
def plot_property_type_price_trends(df_filtered):
    price_trends = df_filtered.groupby(['Property Type', 'Year'])['Price'].mean().unstack(level=0)
    price_trends.columns = price_trends.columns.map(property_type_mapping)
    price_trends.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title('Price Trends for Different Property Types(2007-2017')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.legend(title='Property Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot transaction trends for different property types
def plot_property_type_transaction_trends(df_filtered):
    transaction_counts = df_filtered.groupby(['Property Type', 'Year']).size().unstack(level=0)
    transaction_counts.columns = transaction_counts.columns.map(property_type_mapping)
    transaction_counts.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title('Number of Transactions for Each Property Type(2007-2017)')
    plt.xlabel('Year')
    plt.ylabel('Number of Transactions')
    plt.legend(title='Property Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot seasonal price trends
def plot_seasonal_price_trend(df_filtered):
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

# Preprocess the dataframe
pandarallel.initialize()
df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x: x.split(' ')[0])
df['Year'] = df['Date of Transfer'].str[:4]
df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x: __import__('datetime').datetime.strptime(x, '%Y-%m-%d'))
df['Year'] = df['Date of Transfer'].dt.year
df['Month'] = df['Date of Transfer'].dt.month

# Drop unwanted columns
df.drop(columns='Transaction unique identifier', axis=1, inplace=True)
df.drop(columns='Duration', axis=1, inplace=True)
df.drop(columns='PPDCategory Type', axis=1, inplace=True)
df.drop(columns='Record Status - monthly file only', axis=1, inplace=True)

# Filter data for 2007-2017
df_filtered_2007_2017 = filter_data_by_year(df, 2007, 2017)

# EDA – seasonal transaction trend (2007-2017)
plot_transaction_heatmap(df_filtered_2007_2017)

# EDA- old vs new property price trend(2007-2017)
plot_old_new_property_price_trends(df_filtered_2007_2017)

# EDA- property type price trend(2007-2017)
plot_property_type_price_trends(df_filtered_2007_2017)

# EDA- property type transaction trend(2007-2017)
plot_property_type_transaction_trends(df_filtered_2007_2017)

# Filter data for 2015-2017 for seasonal price trend
df_filtered_2015_2017 = filter_data_by_year(df, 2015, 2017)

# Plot seasonal price trends for 2015-2017
plot_seasonal_price_trend(df_filtered_2015_2017)

# Plot transaction trends for old vs new properties (2007-2017)
plot_old_new_property_transaction_trends(df_filtered_2007_2017)



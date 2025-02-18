# -*- coding: utf-8 -*-
"""
@author: subhashri ravichandran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from pandarallel import pandarallel

# Initialize parallel processing
pandarallel.initialize()

# Define the file path for reading the dataset
FILE_PATH = "C:\\Users\\subha\\OneDrive\\Dokumente\\BCM\\WinterSemester\\Data Science\\Project\\Kaggle\\price_paid_records.csv"

# Define the property type mapping
PROPERTY_TYPE_MAPPING = {
    0: 'Detached',
    1: 'Semi-Detached',
    2: 'Terraced',
    3: 'Flats',
    4: 'Other'
}

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop unwanted columns
    df.drop(columns=['Transaction unique identifier', 'Duration', 'PPDCategory Type', 'Record Status - monthly file only'], axis=1, inplace=True)
    
    # Convert 'Date of Transfer' to datetime object
    df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x: x.split(' ')[0])
    df['Date of Transfer'] = df['Date of Transfer'].parallel_apply(lambda x: __import__('datetime').datetime.strptime(x, '%Y-%m-%d'))
    df['Year'] = df['Date of Transfer'].dt.year
    df['Month'] = df['Date of Transfer'].dt.month
    
    # Map property types to numeric values
    df['Property Type'] = df['Property Type'].map({
        'D': 0,
        'S': 1,
        'T': 2,
        'F': 3,
        'O': 4
    })
    
    return df

# Function to filter data based on years
def filter_data_by_year(df, start_year, end_year):
    return df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Function to train and predict property prices
def train_and_predict_property_prices(df, start_year, end_year, years_to_predict):
    df_filtered = filter_data_by_year(df, start_year, end_year)
    
    # Group by Year and Property Type to calculate average price
    df_avg_price = df_filtered.groupby(['Year', 'Property Type'], as_index=False)['Price'].mean()
    
    # Features and target for modeling
    X = df_avg_price[['Year', 'Property Type']]
    y = df_avg_price['Price']
    
    # Initialize the model
    model = LinearRegression()
    
    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    print(f"Property Price Cross-Validation MAE: {-cv_results.mean()}")
    
    # Train on the entire dataset
    model.fit(X, y)
    
    # Prepare data for forecasting (2017-2022)
    property_types = [0, 1, 2, 3, 4]
    forecast_data = pd.DataFrame([(year, pt) for year in years_to_predict for pt in property_types], columns=['Year', 'Property Type'])
    
    # Make predictions for the years 2017-2022
    forecast_data['Predicted Price'] = model.predict(forecast_data[['Year', 'Property Type']])
    
    return forecast_data

# Function to train and predict transaction counts
def train_and_predict_transaction_counts(df, start_year, end_year, years_to_predict):
    df_filtered = filter_data_by_year(df, start_year, end_year)
    
    # Group by Year and Property Type to calculate transaction counts
    transaction_counts = df_filtered.groupby(['Property Type', 'Year']).size().reset_index(name='Transaction Count')
    
    # Features and target for modeling
    X_trans = transaction_counts[['Year', 'Property Type']]
    y_trans = transaction_counts['Transaction Count']
    
    # Initialize the model
    model_trans = LinearRegression()
    
    # Perform cross-validation for transaction count model
    cv_results_trans = cross_val_score(model_trans, X_trans, y_trans, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"Transaction Count Cross-Validation MAE: {-cv_results_trans.mean()}")
    
    # Train on the entire dataset
    model_trans.fit(X_trans, y_trans)
    
    # Prepare data for forecasting (2017-2022)
    property_types = [0, 1, 2, 3, 4]
    forecast_transaction_data = pd.DataFrame([(year, pt) for year in years_to_predict for pt in property_types], columns=['Year', 'Property Type'])
    
    # Make predictions for the years 2017-2022
    forecast_transaction_data['Predicted Transaction Count'] = model_trans.predict(forecast_transaction_data[['Year', 'Property Type']])
    
    return forecast_transaction_data

# Function to plot predictions (property prices)
def plot_predictions(forecast_data, property_type_mapping, title='Predicted Property Prices'):
    plt.figure(figsize=(10, 6))
    
    for pt in range(5):
        forecast_for_pt = forecast_data[forecast_data['Property Type'] == pt]
        plt.plot(forecast_for_pt['Year'], forecast_for_pt['Predicted Price'], label=property_type_mapping.get(pt, 'Unknown'), linestyle='-', marker='*')
        
        # Annotate the last point
        last_year = forecast_for_pt['Year'].iloc[-1]
        last_price = forecast_for_pt['Predicted Price'].iloc[-1]
        rounded_price = round(last_price)
        plt.text(last_year, last_price, f'{rounded_price}', fontsize=12, ha='center', va='bottom')
    
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.title(title)
    plt.legend(title="Property Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot predictions (transaction counts)
def plot_transaction_predictions(forecast_transaction_data, property_type_mapping, title='Predicted Transaction Counts'):
    plt.figure(figsize=(10, 6))
    
    for pt in range(5):
        forecast_trans_for_pt = forecast_transaction_data[forecast_transaction_data['Property Type'] == pt]
        property_type_name = property_type_mapping.get(pt, 'Unknown')
        plt.plot(forecast_trans_for_pt['Year'], forecast_trans_for_pt['Predicted Transaction Count'], linestyle='-', marker='o', label=property_type_name)
        
        # Annotate the last point
        last_year_trans = forecast_trans_for_pt['Year'].iloc[-1]
        last_count = forecast_trans_for_pt['Predicted Transaction Count'].iloc[-1]
        rounded_count = round(last_count)
        plt.text(last_year_trans, last_count, f'{rounded_count}', fontsize=12, ha='center', va='bottom')
    
    plt.xlabel('Year')
    plt.ylabel('Transaction Count')
    plt.title(title)
    plt.legend(title="Property Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == '__main__':
    # Load and preprocess data
    df = load_and_preprocess_data(FILE_PATH)
    
    # Forecast Property Prices for 2017-2022
    forecast_data = train_and_predict_property_prices(df, 1995, 2015, years_to_predict=np.arange(2017, 2023))
    
    # Plot property price predictions
    plot_predictions(forecast_data, PROPERTY_TYPE_MAPPING, 'Predicted Property Prices (2017-2022)')
    
    # Forecast Transaction Counts for 2017-2022
    forecast_transaction_data = train_and_predict_transaction_counts(df, 1995, 2015, years_to_predict=np.arange(2017, 2023))
    
    # Plot transaction count predictions
    plot_transaction_predictions(forecast_transaction_data, PROPERTY_TYPE_MAPPING, 'Predicted Transaction Counts (2017-2022)')

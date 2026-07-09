# Import Libraries
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import time


import warnings
warnings.filterwarnings('ignore')

# Get the directory of the current file (predict.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define fixed holiday dates
fixed_holidays = {
    'new_year': (1, 1),
    'valentines': (2, 14),
    'christmas': (12, 25)
}

def get_mothers_day(year):
    """Return the date of the second Sunday in May for the given year."""
    may_first = pd.Timestamp(year, 5, 1)
    # Find the first Sunday in May
    first_sunday = may_first + pd.offsets.Week(weekday=6)
    # Mother's Day is the second Sunday in May
    mothers_day = first_sunday + pd.offsets.Week(weekday=6)
    return mothers_day

def get_thanksgiving(year):
    """Return the date of the fourth Thursday in November for the given year."""
    nov_first = pd.Timestamp(year, 11, 1)
    # Find the first Thursday in November
    first_thursday = nov_first + pd.offsets.Week(weekday=3)
    # Thanksgiving is the fourth Thursday in November
    thanksgiving = first_thursday + pd.offsets.Week(weekday=3, n=3)
    return thanksgiving

def get_cyber_monday(year):
    """Return the date of the Monday after Thanksgiving for the given year."""
    thanksgiving = get_thanksgiving(year)
    cyber_monday = thanksgiving + pd.DateOffset(days=4)
    return cyber_monday

def is_holiday(date):
    # Check fixed holidays
    for month, day in fixed_holidays.values():
        holiday_date = pd.Timestamp(date.year, month, day)
        if abs((date - holiday_date).days) <= 7:
            return 1
    
    # Check dynamic holidays
    year = date.year
    
    # Mother's Day
    mothers_day = get_mothers_day(year)
    if abs((date - mothers_day).days) <= 7:
        return 1
    
    # Thanksgiving
    thanksgiving = get_thanksgiving(year)
    if abs((date - thanksgiving).days) <= 7:
        return 1
    
    # Cyber Monday
    cyber_monday = get_cyber_monday(year)
    if abs((date - cyber_monday).days) <= 7:
        return 1
    
    return 0


def preprocess(input_data):
  if 'date' in input_data.columns:
      # Ensure date is in datetime format
      input_data['date'] = pd.to_datetime(input_data['date'])
      
      # Calculate date related parameters
      input_data['month'] = input_data['date'].dt.month
      input_data['weekday'] = input_data['date'].dt.weekday
      input_data['weeknum'] = input_data['date'].dt.isocalendar().week
      input_data['is_weekend'] = input_data['weekday'].isin([5, 6]).astype(int)
      input_data['is_holiday_period'] = input_data['date'].apply(is_holiday).astype(int)
  return input_data

def predict(input_data):
  
# Construct the full path to flights.csv
  path = os.path.join(current_dir, 'pkl_files/')

  # Preprocess and create features for test
  input_data = preprocess(input_data)
  # Feed the distance & time taken by the flight
  load_distance_time = pd.read_csv(os.path.join(current_dir, 'load_distance_time.csv'))
  input_data = pd.merge(input_data, load_distance_time, on=['from', 'to', 'flighttype', 'agency'], how='left')

  # Load the encoder and scaler for use on new data
  with open(path+'encoder.pkl', 'rb') as file:
      loaded_encoder = pickle.load(file)

  with open(path+'scaler.pkl', 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Separate categorical and numerical features
  columns_for_one_hot_encoding = ['from', 'to', 'flighttype', 'agency']
  numerical_features = ['time', 'distance', 'is_holiday_period', 'month', 'weekday', 'weeknum', 'is_weekend']

  # Transform new data using the loaded encoder and scaler
  input_data_cat = loaded_encoder.transform(input_data[columns_for_one_hot_encoding])
  input_data_num = loaded_scaler.transform(input_data[numerical_features])

  input_data = np.hstack((input_data_num, input_data_cat))

  with open(path+'best_xgb_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
  
  y_pred = loaded_model.predict(input_data)

  return y_pred

def generate_combinations():
    # Load flights.csv to get combinations of from, to, flighttype, agency

    df_flights = pd.read_csv(os.path.join(current_dir, 'flights.csv'))  # Adjust path as necessary
    # Change column names to lowercase with underscores instead of spaces
    df_flights.columns = df_flights.columns.str.lower().str.replace(' ', '_')

    # Extract unique combinations of from, to, flighttype, agency
    combinations_df = df_flights[['from', 'to', 'flighttype', 'agency']].drop_duplicates()

    return combinations_df

def batch_predictions():
    try:
        date_today = datetime.today().date()
        combinations_df = generate_combinations()
        
        # Preprocess and generate input data for predictions
        input_data = pd.DataFrame({
            'date': [date_today] * len(combinations_df),
            'from': combinations_df['from'],
            'to': combinations_df['to'],
            'flighttype': combinations_df['flighttype'],
            'agency': combinations_df['agency']
        })
        
        # Predict prices
        predicted_prices = predict(input_data)
        # Create output DataFrame with predicted prices
        df_out = input_data[['date', 'from', 'to', 'flighttype', 'agency']].copy()
        df_out['price'] = predicted_prices
        
        # Ensure predictions.csv exists or create it with proper permissions
        
        file_path = os.path.join(current_dir, 'predictions.csv')
        file_exists = os.path.isfile(file_path)

        if file_exists:
            # Load existing predictions to check for duplicates
            existing_df = pd.read_csv(file_path)
            
            # Convert date columns to datetime64[ns] for consistency
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            df_out['date'] = pd.to_datetime(df_out['date'])
            
            # Merge new predictions with existing ones to update if necessary
            merged_df = pd.merge(existing_df, df_out, on=['date', 'from', 'to', 'flighttype', 'agency'], how='outer', suffixes=('_old', ''))
            
            # Select final rows with updated values, prioritizing new predictions over old ones
            merged_df['price'] = merged_df['price'].fillna(merged_df['price_old'])
            final_df = merged_df[['date', 'from', 'to', 'flighttype', 'agency', 'price']]
        else:
            # If file doesn't exist, write the new predictions directly
            final_df = df_out.copy()
        
        # Append to predictions.csv with correct permissions
        final_df.to_csv(file_path, mode='w', header=True, index=False)
        
        # Set file permissions (adjust as needed) — best-effort; not meaningful
        # for files bind-mounted into a container from a different-UID host.
        try:
            os.chmod(file_path, 0o644)
        except PermissionError:
            pass
        print(f"Appended predictions to {file_path}")

    except Exception as e:
        print(e)
        exit(1)
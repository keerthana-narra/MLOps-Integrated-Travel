# Import Libraries
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Dataset Loading
df = pd.read_csv('../data/flights.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Data Preprocessing
def check_drop_duplicates(df):
    if len(df[df.duplicated()]) > 0:
        print(f'Count of duplicate rows: {len(df[df.duplicated()])}')
        print('Dropping duplicates')
        df = df.drop_duplicates()
    else:
        print('There are no duplicates.')
    return df

def get_mothers_day(year):
    """Return the date of the second Sunday in May for the given year."""
    may_first = pd.Timestamp(year, 5, 1)
    first_sunday = may_first + pd.offsets.Week(weekday=6)
    mothers_day = first_sunday + pd.offsets.Week(weekday=6)
    return mothers_day

def get_thanksgiving(year):
    """Return the date of the fourth Thursday in November for the given year."""
    nov_first = pd.Timestamp(year, 11, 1)
    first_thursday = nov_first + pd.offsets.Week(weekday=3)
    thanksgiving = first_thursday + pd.offsets.Week(weekday=3, n=3)
    return thanksgiving

def get_cyber_monday(year):
    """Return the date of the Monday after Thanksgiving for the given year."""
    thanksgiving = get_thanksgiving(year)
    cyber_monday = thanksgiving + pd.DateOffset(days=4)
    return cyber_monday

def is_holiday(date):
    fixed_holidays = {
        'new_year': (1, 1),
        'valentines': (2, 14),
        'christmas': (12, 25)
    }

    for month, day in fixed_holidays.values():
        holiday_date = pd.Timestamp(date.year, month, day)
        if abs((date - holiday_date).days) <= 7:
            return 1

    year = date.year

    mothers_day = get_mothers_day(year)
    if abs((date - mothers_day).days) <= 7:
        return 1

    thanksgiving = get_thanksgiving(year)
    if abs((date - thanksgiving).days) <= 7:
        return 1

    cyber_monday = get_cyber_monday(year)
    if abs((date - cyber_monday).days) <= 7:
        return 1

    return 0

# Converting datetime columns to same format
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Create new features
df['is_holiday_period'] = df['date'].apply(is_holiday)
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['weeknum'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Drop unnecessary columns
df = df.drop(columns=['travelcode', 'usercode', 'date'])

# Check and drop duplicates
df = check_drop_duplicates(df)

# Data Transformation
# Separate features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Columns for one-hot encoding
columns_for_one_hot_encoding = ['from', 'to', 'flighttype', 'agency']

# Initialize encoder and scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
scaler = MinMaxScaler()

# One-hot encode categorical columns
X_train_cat = encoder.fit_transform(X_train[columns_for_one_hot_encoding])

# Scale numerical columns
numerical_features = ['time', 'distance', 'is_holiday_period', 'month', 'weekday', 'weeknum', 'is_weekend']
X_train_num = scaler.fit_transform(X_train[numerical_features])

# Combine encoded categorical and scaled numerical features
X_train = np.hstack((X_train_num, X_train_cat))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Modelling & Evaluation
# Function to calculate adjusted R²
def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Function to train and evaluate model
# Function to train and evaluate model
def train_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    with mlflow.start_run(run_name=model_name) as run:
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        time_taken = time.time() - start_time

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        adj_r2 = adjusted_r2_score(r2, X_val.shape[0], X_val.shape[1])
        explained_variance = explained_variance_score(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("adjusted_r2", adj_r2)
        mlflow.log_metric("explained_variance", explained_variance)
        mlflow.log_metric("time_taken", time_taken)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

        return {
            'rmse': rmse,
            'r2': r2,
            'adjusted_r2': adj_r2,
            'explained_variance': explained_variance,
            'time_taken': time_taken,
            'model_name': model_name,
            'run_id': run.info.run_id
        }

    
# Train and evaluate models
results = {}

# Linear Regression
lr_model = LinearRegression()
results['Linear Regression'] = train_evaluate_model(lr_model, X_train, y_train, X_val, y_val, "LinearRegression")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
results['Random Forest'] = train_evaluate_model(rf_model, X_train, y_train, X_val, y_val, "RandomForest")

# XGBoost
xgb_model = XGBRegressor()
results['XGBoost'] = train_evaluate_model(xgb_model, X_train, y_train, X_val, y_val, "XGBoost")

# Display results
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
print(results_df)

# Hyperparameter Tuning with XGBoost
# Parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4, 5]
}

# Start an MLflow run for hyperparameter tuning
with mlflow.start_run(run_name="XGBoost Hyperparameter Tuning") as run:
    # Initialize XGBoost model
    xgb_model = XGBRegressor()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=5,  # Number of parameter settings sampled
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all processors
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = random_search.best_params_
    best_xgb_model = random_search.best_estimator_

    # Log best parameters and metrics
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(best_xgb_model, "BestXGBoostModel")

    # Evaluate the best model
    y_pred = best_xgb_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    adj_r2 = adjusted_r2_score(r2, X_val.shape[0], X_val.shape[1])
    explained_variance = explained_variance_score(y_val, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("adjusted_r2", adj_r2)
    mlflow.log_metric("explained_variance", explained_variance)

    # Print evaluation metrics
    print(f"Best parameters found: {best_params}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print(f"Adjusted R²: {adj_r2}")
    print(f"Explained Variance: {explained_variance}")
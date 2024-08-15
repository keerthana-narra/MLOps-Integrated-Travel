# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from gender_guesser.detector import Detector
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import streamlit as st
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

df_hotels = pd.read_csv('datasets/hotels.csv')
df_users = pd.read_csv('datasets/users.csv')
# Change column names to lowercase with underscores instead of spaces
df_hotels.columns = df_hotels.columns.str.lower().str.replace(' ', '_')
df_users.columns = df_users.columns.str.lower().str.replace(' ', '_')

# Merge dataframes
df = pd.merge(df_hotels, df_users, left_on='usercode', right_on='code', how='inner')

# This function helps us to check and drop duplicates whenever required
def check_drop_duplications(df):
  if len(df[df.duplicated()]) > 0:
    print(f'Count of duplicate rows : {len(df[df.duplicated()])}')
    print(f'Dropping duplicates')
    df = df.drop_duplicates()
  return df

def preprocess(df):
    # Converting datetime columns to same format
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

    # Extract first and last names
    df['name_y'] = df['name_y'].apply(lambda x: x.split()[0])
    
    # Assuming df and detector are already defined
    detector = Detector()

    df['gender'] = np.where(df['gender'].isnull(), df['name_y'].apply(lambda x: detector.get_gender(x.split()[0])), df['gender'])
    # Mapping dictionary
    gender_mapping = {
        'andy': 'NA',
        'unknown': 'NA',
        'mostly_male': 'male',
        'mostly_female': 'female',
        'male': 'male',
        'female': 'female'
    }

    # Replace the predicted gender values using the mapping dictionary
    df['gender'] = df['gender'].map(gender_mapping)

    # Lets see if we have any duplicate rows
    df = check_drop_duplications(df)

    # Create Additional Features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['cost_per_day'] = df['total'] / df['days']
    df['price_range'] = pd.cut(df['price'], bins=[0, 100, 300, 500, 1000], labels=['Low', 'Medium', 'High', 'Premium'])
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '56+'])

    # Drop unnecessary columns and columns with very high percentage of nulls
    # Drop columns which are not useful because of high number of missing values or high number of categories
    drop_cols = ['name_y','travelcode']
    df = df.drop(columns=drop_cols)

    # check duplicates after removing date
    df = check_drop_duplications(df)

    le_name = LabelEncoder()
    df['name_encoded'] = le_name.fit_transform(df['name_x'])

    le_place = LabelEncoder()
    df['place_encoded'] = le_place.fit_transform(df['place'])

    le_company = LabelEncoder()
    df['company_encoded'] = le_company.fit_transform(df['company'])

    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])

    # Categorical features to be encoded
    categorical_features = ['place', 'age_group', 'gender', 'company']

    # Create a OneHotEncoder instance
    encoder = OneHotEncoder(sparse=False)

    # Apply one-hot encoding to the categorical features
    encoded_categorical = encoder.fit_transform(df[categorical_features])

    # Combine the encoded categorical features with the numeric features
    numeric_features = df[['days', 'price', 'month', 'day_of_week', 'cost_per_day', 'age']]
    content_features = np.hstack((numeric_features, encoded_categorical))

    user_item_matrix = df.pivot_table(index='usercode', columns='name_encoded', values='total', fill_value=0)

    # Normalize all features
    scaler = MinMaxScaler()
    encoded_features_scaled  = scaler.fit_transform(content_features)

    return df, encoded_features_scaled

df, encoded_features_scaled = preprocess(df)
# Load the collaborative model
svd_model = joblib.load('collaborative_recommendation.pkl')

def get_content_based_recommendations(user_code, df, encoded_features_scaled, top_n=5):
    # Check if the user exists in the data
    if user_code not in df['usercode'].unique():
        print("Caution: New user detected. Using top visited hotels for recommendations.")
        top_hotels = df['name_x'].value_counts().head(top_n).index.tolist()
        return top_hotels

    # Get the index of the user
    user_idx = df[df['usercode'] == user_code].index[0]

    # Get the user's feature vector
    user_features = encoded_features_scaled[user_idx].reshape(1, -1)

    # Compute the similarity between the user and all hotels
    similarities = cosine_similarity(user_features, encoded_features_scaled)

    # Get the indices of the most similar hotels
    similar_indices = similarities.argsort()[0][::-1][:top_n]

    # Retrieve the corresponding hotel names
    similar_hotels = df['name_x'].iloc[similar_indices].tolist()  # Use .tolist() to ensure it's a list of individual names

    return similar_hotels

def hybrid_recommendations(user_code, svd_model, df, encoded_features_scaled, top_n=5):
    
    # Step 1: Collaborative filtering
    hotel_ids = df['name_x'].unique()
    if user_code in df['usercode'].unique():
        collaborative_predictions = [svd_model.predict(user_code, hotel_id) for hotel_id in hotel_ids]
        collaborative_predictions.sort(key=lambda x: x.est, reverse=True)
        top_collaborative = [cp.iid for cp in collaborative_predictions[:top_n]]
        top_collaborative_hotels = [df[df['name_x'].str.contains(hotel)].iloc[0]['name_x'] for hotel in top_collaborative]  # Ensure these are individual names
    else:
        print("Caution: New user detected. Using top visited hotels for recommendations.")
        return df['name_x'].value_counts().head(top_n).index.tolist()

    # Step 2: Content-based filtering
    top_hotels = get_content_based_recommendations(user_code, df, encoded_features_scaled, top_n)

    # Combine both and remove duplicates
    combined_recommendations = list(dict.fromkeys(top_collaborative_hotels + top_hotels))[:top_n]

    return combined_recommendations








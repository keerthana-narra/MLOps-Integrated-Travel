from import_lib import *

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
  # Preprocess and create features for test
  input_data = preprocess(input_data)

  # Load the encoder and scaler for use on new data
  with open('encoder.pkl', 'rb') as file:
      loaded_encoder = pickle.load(file)

  with open('scaler.pkl', 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Separate categorical and numerical features
  columns_for_one_hot_encoding = ['from', 'to', 'flighttype', 'agency']
  numerical_features = ['time', 'distance', 'is_holiday_period', 'month', 'weekday', 'weeknum', 'is_weekend']

  # Transform new data using the loaded encoder and scaler
  input_data_cat = loaded_encoder.transform(input_data[columns_for_one_hot_encoding])
  input_data_num = loaded_scaler.transform(input_data[numerical_features])

  input_data = np.hstack((input_data_num, input_data_cat))

  with open('best_xgb_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
  
  y_pred = loaded_model.predict(input_data)

  return y_pred
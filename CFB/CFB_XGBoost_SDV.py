import os
import pandas as pd
from sportsdataverse.cfb import load_cfb_pbp
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import numpy as np

# File to save the raw loaded data
raw_data_file = 'cfb_pbp_raw_data.pkl'

# Check if the raw data is already saved locally
if os.path.exists(raw_data_file):
    # Load the raw data from the file
    with open(raw_data_file, 'rb') as f:
        cfb_pbp = pickle.load(f)
    print("Raw data loaded from file.")
else:
    # Load the data for specific seasons
    cfb_pbp = load_cfb_pbp(seasons=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])

    # Convert Polars DataFrame to Pandas DataFrame
    cfb_pbp = cfb_pbp.to_pandas()

    # Save the raw data to a file for future use
    with open(raw_data_file, 'wb') as f:
        pickle.dump(cfb_pbp, f)
    print("Raw data processed and saved to file.")

# Columns of interest
columns_of_interest = ['wpa', 'wp_before', 'start.ExpScoreDiff', 'start.TimeSecsRem', 'start.distance', 
                       'qbr_epa', 'start.yardLine', 'clock.displayValue', 'start.yardsToEndzone', 'pos_score_diff', 
                       'start.down', 'EPA', 'playType']

cfb_pass_rush = cfb_pbp[columns_of_interest].copy()

# Filter for pass or rush plays
cfb_pass_rush = cfb_pass_rush[cfb_pass_rush['playType'].str.contains('Rush|Pass', case=False)]

# Map 'Rush' to 0 and 'Pass' to 1
cfb_pass_rush['playType'] = cfb_pass_rush['playType'].map(lambda x: 0 if 'Rush' in x else 1)

# Function to convert 'MM:SS' to total seconds
def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None  # Handle NaN values
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

# Apply the conversion function to the relevant column
cfb_pass_rush['clock.displayValue'] = cfb_pass_rush['clock.displayValue'].apply(convert_time_to_seconds)

# Replace infinite values caused by invalid divisions
def safe_divide(numerator, denominator):
    return np.where(denominator != 0, numerator / denominator, 0)

# Feature engineering: Add interaction terms and derived features
cfb_pass_rush['yardLine_distance_interaction'] = cfb_pass_rush['start.yardLine'] * cfb_pass_rush['start.distance']

# Replace any invalid divisions by zero with a safe alternative
cfb_pass_rush['time_to_endzone_ratio'] = safe_divide(cfb_pass_rush['start.TimeSecsRem'], cfb_pass_rush['start.yardsToEndzone'])

# Drop missing values and ensure no infinite values remain
cfb_pass_rush.replace([np.inf, -np.inf], np.nan, inplace=True)
cfb_pass_rush.dropna(inplace=True)

# Split data into features (X) and target (y)
y = cfb_pass_rush['playType']
X = cfb_pass_rush.drop(columns=['playType'])

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost model with hyperparameter tuning
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.5]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Output best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Refit the model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Model Evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))

# Cross-Validation for robust accuracy estimate
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean()}')

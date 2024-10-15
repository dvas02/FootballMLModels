import os
import pandas as pd
from sportsdataverse.cfb import load_cfb_pbp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


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
  # NOTE There seems to be an error with 2022
  cfb_pbp = load_cfb_pbp(seasons=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
  '''for season in range(2006, 2023):  # Adjust the range according to your needs
    try:
        cfb_pbp = load_cfb_pbp(seasons=[season])
        #print(f"Columns for {season}: {cfb_pbp.columns}")
    except Exception as e:
        print(f"Error loading data for {season}: {e}")'''

  # Convert Polars DataFrame to Pandas DataFrame
  cfb_pbp = cfb_pbp.to_pandas()


  # Save the raw data to a file for future use
  with open(raw_data_file, 'wb') as f:
    pickle.dump(cfb_pbp, f)
  print("Raw data processed and saved to file.")
    
    
# Show all the possible columns
#for i in cfb_pbp.columns:
#  print(i)
  
#######################
# Data Preprocessing  #
#######################
# Filter relevant columns related to pass rush

# Other potential columns: qb_epa, xyac_epa, xyac_mean_yardage, xyac_median_yardage, xyac_success, xyac_fd, season_type

#Columns taken:
### start.yardLine -> location of the ball on the field, measured in yards from the opponent's end zone
### clock.displayValue -> time remaining in the quarter
### start.TimeSecsRem -> time remaining in game
### start.yardsToEndzone -> yards to go for a 1st down
### pos_score_diff -> score difference between two teams
### start.down -> 1st - 4th down
### period.number -> 1 - 4 quarter - REMOVED noise --> precision went down 0.02%
### playType -> target variable (pass or rush)
### EPA
### qbr_epa
### start.distance
### half XXX
### start.TimeSecsRem
### end_of_half XXX
### is_home XXX
### middle_8 XXX
### start.ExpScoreDiff
### wp_before
### wpa

columns_of_interest = ['wpa', 'wp_before', 'start.ExpScoreDiff', 'start.TimeSecsRem', 'start.distance', 'period.number', 'qbr_epa', 'start.yardLine', 'clock.displayValue', 'start.TimeSecsRem', 'start.yardsToEndzone', 'pos_score_diff', 'start.down', 'EPA', 'playType']

#for i in cfb_pbp.columns:
#    print(i)

#print(cfb_pbp['playType'])

cfb_pass_rush = cfb_pbp[columns_of_interest].copy()


# Filter for pass or rush plays
#cfb_pass_rush = cfb_pass_rush[cfb_pass_rush['playType'].isin(['Pass', 'Rush'])]
# Filter for pass or rush plays using str.contains
#cfb_pass_rush = cfb_pass_rush[cfb_pass_rush['playType'].str.contains('Rush|Pass', case=False)]


#print(cfb_pass_rush.columns)

# Drop missing values
#cfb_pass_rush.dropna(inplace=True)
#print(cfb_pass_rush)



# Feature Engineering
# Convert categorical features to numerical using get_dummies or other encoding methods

# Convert 'rush' to 0 and 'pass' to 1
#cfb_pass_rush['playType'] = cfb_pass_rush['playType'].map({'Rush': 0, 'Pass': 1})
# Convert 'Rush' to 0 and 'Pass' to 1 using mapping
cfb_pass_rush['playType'] = cfb_pass_rush['playType'].map(lambda x: 0 if 'Rush' in x else 1 if 'Pass' in x else None)

#print(cfb_pass_rush['playType'])
#print(cfb_pass_rush)



#print(cfb_pass_rush)

# Function to convert 'MM:SS' to total seconds
def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None  # Handle NaN values
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

# Apply the conversion function to the relevant column
cfb_pass_rush['clock.displayValue'] = cfb_pass_rush['clock.displayValue'].apply(convert_time_to_seconds)






cfb_pass_rush.dropna(inplace=True)



# Split data into features (X) and target (y)
# Assign the target variable 'y' as the 'playType' column
y = cfb_pass_rush['playType']

# Assign the features 'X' as all columns except 'playType'
X = cfb_pass_rush.drop(columns=['playType'])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Model Training
# Using RandomForestClassifier as a first try
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)



# Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Output feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

# Print best to worst labels
print(feature_importances.sort_values(ascending=False))

# Print worst to best labels
#print(feature_importances.sort_values())
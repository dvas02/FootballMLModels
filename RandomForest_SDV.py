import os
import pandas as pd
from sportsdataverse.nfl import load_nfl_pbp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


# File to save the raw loaded data
raw_data_file = 'nfl_pbp_raw_data.pkl'


# Check if the raw data is already saved locally
if os.path.exists(raw_data_file):
  # Load the raw data from the file
  with open(raw_data_file, 'rb') as f:
      nfl_pbp = pickle.load(f)
  print("Raw data loaded from file.")
else:
  # Load the data for specific seasons
  nfl_pbp = load_nfl_pbp(seasons=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

  # Convert Polars DataFrame to Pandas DataFrame
  nfl_pbp = nfl_pbp.to_pandas()

  # Save the raw data to a file for future use
  with open(raw_data_file, 'wb') as f:
    pickle.dump(nfl_pbp, f)
  print("Raw data processed and saved to file.")
    
    
# Show all the possible columns
#for i in nfl_pbp.columns:
#  print(i)
  
#######################
# Data Preprocessing  #
#######################
# Filter relevant columns related to pass rush

# Other potential columns: qb_epa, xyac_epa, xyac_mean_yardage, xyac_median_yardage, xyac_success, xyac_fd, season_type

#Columns taken:
# yardline_100 -> location of the ball on the field, measured in yards from the opponent's end zone
# quarter_seconds_remaining -> time remaining in the quarter
# game_seconds_remaining -> time remaining in game
# ydstogo -> yards to go for a 1st down
# score_differential -> score difference between two teams
# down -> 1st - 4th down
# qtr -> 1 - 4 quarter - REMOVED noise --> precision went down 0.02%
# weather - REMOVED since its not quantitative
# wind - REMOVED noise -->
# drive_inside20 - REMOVED noise --> precision down 0.01%
# play_type -> target variable (pass or rush)

columns_of_interest = ['qtr', 'drive_inside20', 'qb_epa', 'wind', 'yardline_100', 'quarter_seconds_remaining', 'game_seconds_remaining', 'ydstogo', 'score_differential', 'down', 'epa', 'play_type']

#for i in nfl_pbp.columns:
#    print(i)


nfl_pass_rush = nfl_pbp[columns_of_interest].copy()


# Filter for pass or rush plays
nfl_pass_rush = nfl_pass_rush[nfl_pass_rush['play_type'].isin(['pass', 'run'])]


#print(nfl_pass_rush.columns)

# Drop missing values
nfl_pass_rush.dropna(inplace=True)
#print(nfl_pass_rush)



# Feature Engineering
# Convert categorical features to numerical using get_dummies or other encoding methods

# Convert 'rush' to 0 and 'pass' to 1
nfl_pass_rush['play_type'] = nfl_pass_rush['play_type'].map({'run': 0, 'pass': 1})

#print(nfl_pass_rush['play_type'])
#print(nfl_pass_rush)



#print(nfl_pass_rush)




# Split data into features (X) and target (y)
# Assign the target variable 'y' as the 'play_type' column
y = nfl_pass_rush['play_type']

# Assign the features 'X' as all columns except 'play_type'
X = nfl_pass_rush.drop(columns=['play_type'])


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
#print(feature_importances.sort_values(ascending=False))

# Print worst to best labels
print(feature_importances.sort_values())
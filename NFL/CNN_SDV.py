import os
import pandas as pd
from sportsdataverse.nfl import load_nfl_pbp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

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

#######################
# Data Preprocessing  #
#######################
# Columns of interest
columns_of_interest = ['qtr', 'drive_inside20', 'qb_epa', 'wind', 'yardline_100', 'quarter_seconds_remaining', 'game_seconds_remaining', 'ydstogo', 'score_differential', 'down', 'epa', 'play_type']

# Filter for relevant columns
nfl_pass_rush = nfl_pbp[columns_of_interest].copy()

# Filter for pass or rush plays
nfl_pass_rush = nfl_pass_rush[nfl_pass_rush['play_type'].isin(['pass', 'run'])]

# Drop missing values
nfl_pass_rush.dropna(inplace=True)

# Feature Engineering
# Convert 'rush' to 0 and 'pass' to 1
nfl_pass_rush['play_type'] = nfl_pass_rush['play_type'].map({'run': 0, 'pass': 1})

# Split data into features (X) and target (y)
y = nfl_pass_rush['play_type']
X = nfl_pass_rush.drop(columns=['play_type'])

# Normalize the features
X = (X - X.mean()) / X.std()

# Reshape X for CNN (samples, timesteps, features)
X = X.values.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model Training
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 classes: pass and run

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Model Evaluation
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert y_test back to single labels
y_test_single = np.argmax(y_test, axis=1)

print(classification_report(y_test_single, y_pred))

# Since feature importance is not straightforward with CNNs, we skip that part.

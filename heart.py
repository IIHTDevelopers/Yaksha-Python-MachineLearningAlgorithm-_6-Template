import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json


# 1. Load dataset
def load_heart_disease_data():
    # TODO: 
    # 1. Use pandas to read the CSV file "heart.csv"
    # 2. Limit to 303 records to match test expectations
    # 3. Print a message showing the number of records loaded
    # 4. Return the dataframe containing the heart disease data
    df = pd.DataFrame()
    return df


# 2. Preprocess data
def preprocess_heart_data(df):
    # TODO: 
    # 1. Separate features (X) by dropping the "target" column
    # 2. Extract the target variable (y)
    # 3. Print a message indicating features and target are separated
    # 4. Return X and y
    pass
    return X, y


# 3. Split the data
def split_heart_data(X, y, test_size=0.2):
    # TODO: 
    # 1. Use train_test_split to split the data into training and testing sets
    # 2. Use random_state=42 for reproducibility
    # 3. Use the provided test_size parameter
    # 4. Print the sizes of the training and testing sets
    # 5. Return X_train, X_test, y_train, y_test
    pass
    return X_train, X_test, y_train, y_test


def create_train_save_load_model(X_train, y_train, n_estimators=100, max_depth=None,
                                 filename="random_forest_heart_model.pkl"):
    # TODO: 
    # 1. Create a RandomForestClassifier with the provided parameters and random_state=42
    # 2. Train the model using the training data (X_train, y_train)
    # 3. Print a message indicating the model has been trained
    # 4. Save the trained model to the specified filename using joblib.dump()
    # 5. Print a message indicating the model has been saved
    # 6. Load the model from the file using joblib.load()
    # 7. Print a message indicating the model has been loaded
    # 8. Return the loaded model
    pass
    model = RandomForestClassifier()
    return model


# 6. Predict using model only (no manual checking)
def check_new_data_from_json(model, json_file="heart_data.json"):
    # TODO: 
    # 1. Load data from the JSON file
    # 2. Extract patient data from the loaded JSON
    # 3. Create a dataframe with the patient data
    # 4. Make a prediction using the model
    # 5. Print the prediction result (Diseased or Healthy)
    # 6. Return the prediction (0 for Healthy, 1 for Diseased)
    pass
    return -1


# --- Pipeline Execution ---
# TODO: Implement the pipeline execution by calling the functions above in sequence
# 1. Load the dataset using load_heart_disease_data()
# 2. Preprocess the data using preprocess_heart_data()
# 3. Split the data using split_heart_data()
# 4. Create, train, save, and load the model using create_train_save_load_model()
# 5. Check new data from JSON using check_new_data_from_json()

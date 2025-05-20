import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np


# 1. Load chicken disease dataset
def load_chicken_disease_data():
    # TODO: 
    # 1. Use pandas to read the CSV file "chicken_disease_data.csv"
    # 2. Print a message showing the number of records loaded
    # 3. Return the dataframe containing the chicken disease data
    pass
    return pd.DataFrame()


# 2. EDA Function to count chickens with age > 2
def perform_eda_on_age(df):
    # TODO: 
    # 1. Check if 'Age' column exists in the dataframe
    # 2. If it exists, count chickens with age > 2
    # 3. If 'Age' column doesn't exist, print appropriate message
    # 4. Return the count of chickens with age > 2
    count = 0
    return count   
    
# 3. Preprocess data with explicit label encoding
def preprocess_chicken_data(df):
    # TODO: 
    # 1. Check if "Disease Predicted" column exists
    # 2. Clean and normalize values in "Disease Predicted" column
    # 3. Map "Healthy" to 0 and "Sick" to 1 in a new "target" column
    # 4. Check for unmapped labels
    # 5. Drop the original "Disease Predicted" column
    # 6. One-hot encode categorical features
    # 7. Separate features (X) and target (y)
    # 8. Return X, y, and the encoded dataframe
    pass
     return X, y, encoded_df


# 4. Split data
def split_chicken_data(X, y, test_size=0.2):
    # TODO: 
    # 1. Use train_test_split to split the data into training and testing sets
    # 2. Use random_state=42 for reproducibility
    # 3. Use the provided test_size parameter
    # 4. Print the sizes of the training and testing sets
    # 5. Return X_train, X_test, y_train, y_test
    pass
    return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)


# 5. Train model
def create_and_train_model(X_train, y_train):
    # TODO: 
    # 1. Create a DecisionTreeClassifier with random_state=42
    # 2. Train the model using the training data (X_train, y_train)
    # 3. Print a message indicating the model has been trained
    # 4. Return the trained model
    pass
    return DecisionTreeClassifier()


# 6. Predict from new JSON data
def check_new_data_from_json(model, json_file="chicken_data.json"):
    # TODO: 
    # 1. Load data from the JSON file
    # 2. Extract chicken data from the loaded JSON
    # 3. Load the original dataset for preprocessing reference
    # 4. Create a temporary dataframe with the new chicken data
    # 5. Combine with original data for consistent preprocessing
    # 6. Preprocess the combined data
    # 7. Extract the features for the new chicken
    # 8. Make a prediction using the model
    # 9. Print the prediction result (Healthy or Sick)
    # 10. Return the prediction (0 for Healthy, 1 for Sick)
    pass
    return -1


# --- Pipeline Execution ---
# TODO: Implement the pipeline execution by calling the functions above in sequence
# 1. Load the dataset using load_chicken_disease_data()
# 2. Perform EDA using perform_eda_on_age()
# 3. Preprocess the data using preprocess_chicken_data()
# 4. Split the data using split_chicken_data()
# 5. Create and train the model using create_and_train_model()
# 6. Save the model using joblib.dump()
# 7. Predict from JSON using check_new_data_from_json()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np


# 1. Load synthetic chicken disease dataset
def load_chicken_disease_data():
    """
    TODO: Load the chicken disease dataset from the CSV file 'chicken_disease_data.csv'
    
    Returns:
        pandas.DataFrame: The loaded chicken disease dataset (limited to 1000 rows)
    """
    pass


# 2. EDA Function to count chickens with age > 2
def perform_eda_on_age(df):
    """
    TODO: Perform exploratory data analysis on the Age column
    
    This function should count and print the number of chickens with age > 2
    
    Args:
        df (pandas.DataFrame): The chicken disease dataset
    """
    pass


# 3. Preprocess data (Categorical to numerical conversion)
def preprocess_chicken_data(df):
    """
    TODO: Preprocess the chicken disease data by converting categorical features to numerical
    
    This function should:
    1. Convert categorical features to dummy variables
    2. Separate features (X) and target (y) - the target is 'Disease Predicted_Healthy'
    
    Args:
        df (pandas.DataFrame): The chicken disease dataset
        
    Returns:
        tuple: (X, y, df_encoded) where:
            - X contains features
            - y contains the target variable 'Disease Predicted_Healthy'
            - df_encoded is the fully encoded dataframe
    """
    pass


# 4. Split the data
def split_chicken_data(X, y, test_size=0.2):
    """
    TODO: Split the data into training and testing sets
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        test_size (float): Proportion of data to use for testing (default: 0.2)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Split data
    """
    pass


# 5. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    """
    TODO: Create and train a Decision Tree model
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        DecisionTreeClassifier: The trained model
    """
    pass


# 6. Calculate entropy of the target column
def calculate_entropy(y):
    """
    TODO: Calculate the entropy of the target column
    
    Args:
        y (pandas.Series): The target variable
    
    Returns:
        float: The calculated entropy value
    """
    pass


# 7. Check prediction for new data from JSON
def check_new_data_from_json(model, df_encoded, json_file="chicken_data.json"):
    """
    TODO: Make predictions on new chicken data from a JSON file
    
    This function should:
    1. Load chicken data from the JSON file
    2. Process the data to match the encoded format used for training
    3. Make a prediction using the model
    4. Validate the prediction against symptoms
    5. Determine disease type if the chicken is predicted to be diseased
    6. Print the results including:
       - Chicken input data
       - Model prediction
       - Adjusted prediction (if necessary)
       - Disease type (if diseased)
       - Final diagnosis
    
    Args:
        model: The trained model to use for prediction
        df_encoded (pandas.DataFrame): The encoded dataframe used for training
        json_file (str): Path to the JSON file containing chicken data (default: "chicken_data.json")
    """
    pass


# --- Pipeline Execution ---
# TODO: Implement the pipeline to:
# 1. Load the chicken disease data
# 2. Perform EDA on age
# 3. Preprocess the data
# 4. Split the data
# 5. Create and train a model
# 6. Save the model
# 7. Calculate entropy of the target
# 8. Check new data from JSON

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Load synthetic chicken disease dataset
def load_chicken_disease_data():
    """
    Load the chicken disease dataset from the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # TODO: Load the dataset from 'chicken_disease_data.csv'
    # TODO: Return the dataframe
    pass


# 2. Preprocess data (Categorical to numerical conversion)
def preprocess_data(df):
    """
    Preprocess the data by converting categorical features to numerical.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        
    Returns:
        tuple: (X, y, df_encoded) where X is features, y is target, and df_encoded is the encoded dataframe
    """
    # TODO: Convert categorical features to dummy variables
    # TODO: Separate features (X) and target (y) - target should be "Disease Predicted_Healthy"
    # TODO: Return X, y, and the encoded dataframe
    pass


# 3. Split the data
def split_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        test_size (float): Proportion of the dataset to include in the test split
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Split the data into training and testing sets
    # TODO: Use random_state=42 for reproducibility
    # TODO: Return X_train, X_test, y_train, y_test
    pass


# 4. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    """
    Create and train a Decision Tree model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        DecisionTreeClassifier: The trained model
    """
    # TODO: Create a DecisionTreeClassifier with random_state=42
    # TODO: Train the model on X_train and y_train
    # TODO: Return the trained model
    pass

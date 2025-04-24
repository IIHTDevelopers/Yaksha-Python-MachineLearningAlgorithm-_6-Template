import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load heart disease dataset
def load_heart_disease_data():
    """
    Load the heart disease dataset from the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # TODO: Load the dataset from 'heart.csv'
    # TODO: Return the dataframe
    pass


# 2. Preprocess data
def preprocess_data(df):
    """
    Preprocess the data by separating features and target.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    # TODO: Separate features (X) and target (y) - target column is "target"
    # TODO: Return X and y
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


# 4. Create model
def create_model(n_estimators=100, max_depth=None):
    """
    Create a Random Forest model.
    
    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        
    Returns:
        RandomForestClassifier: The created model
    """
    # TODO: Create a RandomForestClassifier with the given parameters
    # TODO: Use random_state=42 for reproducibility
    # TODO: Return the model
    pass


# 5. Train model
def train_model(model, X_train, y_train):
    """
    Train the model on the training data.
    
    Args:
        model: The model to train
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        The trained model
    """
    # TODO: Train the model on X_train and y_train
    # TODO: Return the trained model
    pass


# 6. Save model
def save_model(model, filename="random_forest_heart_model.pkl"):
    """
    Save the model to a file.
    
    Args:
        model: The model to save
        filename (str): The filename to save the model to
    """
    # TODO: Save the model to the specified filename using joblib.dump
    pass

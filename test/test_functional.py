import unittest
import os
import sys
import pandas as pd
import numpy as np
import json

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import chicken
import heart

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import TestUtils directly since we're already in the test directory
from test.TestUtils import TestUtils

class TestHeartDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = heart.load_heart_disease_data()
        import joblib
        self.model = joblib.load("random_forest_heart_model.pkl")
        self.df_encoded = pd.DataFrame([{
            "age": 95, "sex": 1, "cp": 3, "trestbps": 150, "chol": 280, "fbs": 1,
            "restecg": 1, "thalach": 100, "exang": 1, "oldpeak": 3.2, "slope": 2, "ca": 2, "thal": 3
        }])

    def test_load_heart_disease_data(self):
        try:
            expected_columns = {
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak",
                "slope", "ca", "thal", "target"
            }
            df = heart.load_heart_disease_data()  # Replace 'heart' with actual module if needed
            actual_columns = set(df.columns)
            result = expected_columns.issubset(actual_columns)

            self.test_obj.yakshaAssert("HeartDisease TestHeartColumnsPresent", result, "functional")
            print("HeartDisease TestHeartColumnsPresent = Passed" if result else "HeartDisease TestHeartColumnsPresent = Failed")
        except Exception:
            self.test_obj.yakshaAssert("HeartDisease TestHeartColumnsPresent", False, "functional")
            print("HeartDisease TestHeartColumnsPresent = Failed")

    def test_preprocessing_separates_target(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            if not X.empty and not y.empty and "target" not in X.columns:
                self.test_obj.yakshaAssert("HeartDisease TestPreprocessingSeparatesTarget", True, "functional")
                print("HeartDisease TestPreprocessingSeparatesTarget = Passed")
            else:
                self.test_obj.yakshaAssert("HeartDisease TestPreprocessingSeparatesTarget", False, "functional")
                print("HeartDisease TestPreprocessingSeparatesTarget = Failed")
        except Exception:
            self.test_obj.yakshaAssert("HeartDisease TestPreprocessingSeparatesTarget", False, "functional")
            print("HeartDisease TestPreprocessingSeparatesTarget = Failed" )

    def test_split_data_counts(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            X_train, X_test, y_train, y_test = heart.split_heart_data(X, y)
            if len(X_train) == 242 and len(X_test) == 61:
                self.test_obj.yakshaAssert("HeartDisease TestSplitDataCounts", True, "functional")
                print("HeartDisease TestSplitDataCounts = Passed")
            else:
                self.test_obj.yakshaAssert("HeartDisease TestSplitDataCounts", False, "functional")
                print("HeartDisease TestSplitDataCounts = Failed")
        except Exception:
            self.test_obj.yakshaAssert("HeartDisease TestSplitDataCounts", False, "functional")
            print("HeartDisease TestSplitDataCounts = Failed")

    import os

    def test_model_created_trained_saved_loaded_successfully(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)

            # Call the function that saves the model as 'random_forest_heart_model.pkl'
            trained_model = heart.create_train_save_load_model(X, y, n_estimators=10)

            # Check if model is valid and file exists
            file_exists = os.path.exists("random_forest_heart_model.pkl")

            if trained_model and hasattr(trained_model, "predict") and hasattr(trained_model, "fit") and file_exists:
                self.test_obj.yakshaAssert("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully", True, "functional")
                print("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully = Passed")
            else:
                self.test_obj.yakshaAssert("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully", False, "functional")
                print("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully = Failed")
        except Exception:
            self.test_obj.yakshaAssert("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully", False, "functional")
            print("HeartDisease TestModelCreatedTrainedSavedLoadedSuccessfully = Failed")

    def test_heart_is_diseased(self):
        try:
            # Call the function to check the prediction for new patient data from the JSON file
            result = heart.check_new_data_from_json(self.model, json_file="heart_data.json")

            if result == 1:  # 1 means Diseased
                self.test_obj.yakshaAssert("HeartDisease TestHeartIsDiseased", True, "functional")
                print("HeartDisease TestHeartIsDiseased = Passed")
            else:
                self.test_obj.yakshaAssert("HeartDisease TestHeartIsDiseased", False, "functional")
                print(" HeartDisease TestHeartIsDiseased = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("HeartDisease TestHeartIsDiseased", False, "functional")
            print(f"HeartDisease TestHeartIsDiseased = Failed: {str(e)}")

# Import TestUtils directly
class TestChickenDiseaseModel(unittest.TestCase):

    def setUp(self):
        self.test_obj = TestUtils()
        self.df = pd.read_csv("chicken_disease_data.csv")  # Adjust as needed
        self.df_encoded = chicken.preprocess_chicken_data(self.df)[2]

    def test_eda_on_age_count(self):
        try:
            # Call the perform_eda_on_age function
            result = chicken.perform_eda_on_age(self.df)
            
            # Check if the result matches the expected count
            expected_count = 493  # The expected count of chickens with age > 2
            if result == expected_count:
                self.test_obj.yakshaAssert("chickendisease TestEDAOnAgeCount", True, "functional")
                print("chickendisease  TestEDAOnAgeCount = Passed")
            else:
                self.test_obj.yakshaAssert("chickendisease TestEDAOnAgeCount", False, "functional")
                print("chickendisease  TestEDAOnAgeCount = Failed")
        except Exception:
            self.test_obj.yakshaAssert("chickendisease TestEDAOnAgeCount", False, "functional")
            print("chickendisease TestEDAOnAgeCount = Failed")

    def test_preprocessing_output(self):
        try:
            X, y, df_encoded = chicken.preprocess_chicken_data(self.df)

            condition = (
                    not X.empty and
                    not y.empty and
                    set(y.unique()).issubset({0, 1}) and  # Target is binary
                    len(X) == len(y)  # Matching number of rows
            )

            self.test_obj.yakshaAssert("chickendisease TestPreprocessingOutput", condition, "functional")
            print("chickendisease TestPreprocessingOutput = Passed" if condition else "chickendisease TestPreprocessingOutput = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("chickendisease TestPreprocessingOutput", False, "functional")
            print(f"chickendisease  TestPreprocessingOutput = Failed with error: {e}")

    def test_split_chicken_data(self):
        try:
            # Get X and y from the preprocessed data
            X, y, _ = chicken.preprocess_chicken_data(self.df)
            # Call the function to split the data
            X_train, X_test, y_train, y_test = chicken.split_chicken_data(X, y, test_size=0.2)

            # Check if the split sizes are correct (80/20 split)
            total_size = len(X)
            expected_train_size = int(total_size * 0.8)  # 80% for training
            expected_test_size = total_size - expected_train_size  # 20% for testing

            # Assertions to check the correct split
            if len(X_train) == expected_train_size and len(X_test) == expected_test_size and \
                    len(y_train) == expected_train_size and len(y_test) == expected_test_size:
                self.test_obj.yakshaAssert("chickendisease TestSplitChickenData", True, "functional")
                print("chickendisease TestSplitChickenData = Passed")
            else:
                self.test_obj.yakshaAssert("chickendisease TestSplitChickenData", False, "functional")
                print("chickendisease TestSplitChickenData = Failed")
        except Exception as e:
            print(f"Error occurred: {e}")
            self.test_obj.yakshaAssert("chickendisease TestSplitChickenData", False, "functional")
            print("chickendisease TestSplitChickenData = Failed")

    
    def test_load_chicken_disease_data(self):
        try:
            expected_columns = {
                "Age",
                "Breed",
                "Temperature",
                "Eating Behavior",
                "Coughing",
                "Feces Appearance",
                "Water Consumption",
                "Disease Predicted"
            }
            df = chicken.load_chicken_disease_data()
            actual_columns = set(df.columns)
            result = expected_columns.issubset(actual_columns)

            self.test_obj.yakshaAssert("chickendisease TestColumnsPresent", result, "functional")
            print("chickendisease TestColumnsPresent = Passed" if result else "TestColumnsPresent = Failed")
        except Exception:
            self.test_obj.yakshaAssert("chickendisease TestColumnsPresent", False, "functional")
            print("chickendisease TestColumnsPresent = Failed")


class TestChickenModelPrediction(unittest.TestCase):

    def setUp(self):
        self.test_obj = TestUtils()
        import joblib
        self.model = joblib.load("decision_tree_chicken_disease_model.pkl")
        self.df = pd.read_csv("chicken_disease_data.csv")
        self.df_encoded = chicken.preprocess_chicken_data(self.df)[2]

    def test_chicken_is_diseased(self):
        try:
            prediction = chicken.check_new_data_from_json(self.model, json_file="chicken_data.json")

            if prediction == 1:
                self.test_obj.yakshaAssert("chickendisease TestChickenIsDiseased", True, "functional")
                print("chickendisease TestChickenIsDiseased = Passed")
            else:
                self.test_obj.yakshaAssert("chickendisease TestChickenIsDiseased", False, "functional")
                print("chickendisease TestChickenIsDiseased = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("chickendisease TestChickenIsDiseased", False, "functional")
            print(f"chickendisease TestChickenIsDiseased = Failed: {str(e)}")

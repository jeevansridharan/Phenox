"""
PhoenixAI - MVP Model Training Script

This script represents the initial stage of the PhoenixAI MLOps project.
It handles:
1. Loading the built-in Iris dataset.
2. Exploring the dataset (displaying shape, features, classes).
3. Splitting the data into training and testing sets.
4. Training a simple Random Forest Classifier.
5. Evaluating the model's accuracy.
6. Saving the trained model for future use.
"""

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # ---------------------------------------------------------
    # Step 1: Load the dataset
    # ---------------------------------------------------------
    print("Loading Iris dataset...")
    # Load the built-in Iris dataset from Scikit-Learn
    iris = load_iris()
    
    # Create a pandas DataFrame for easier viewing and manipulation
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Display clear console output indicating successful loading
    print("Dataset loaded successfully.\n")
    
    # ---------------------------------------------------------
    # Step 2: Display dataset information
    # ---------------------------------------------------------
    print("--- Dataset Information ---")
    
    # Display the dataset in a simple table (first few rows)
    print("Sample data (first 5 rows):")
    print(df.head().to_string())
    print("\n")
    
    # Show Number of rows and Number of columns
    num_rows, num_cols = df.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    
    # Show Feature names
    print(f"Feature names: {iris.feature_names}")
    
    # Show Target classes
    print(f"Target classes: {list(iris.target_names)}\n")
    
    # ---------------------------------------------------------
    # Step 3: Split the dataset into training and testing sets
    # ---------------------------------------------------------
    # We use 80% of the data for training and 20% for testing to evaluate performance on unseen data
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split into training and testing sets.\n")
    
    # ---------------------------------------------------------
    # Step 4: Train a simple Random Forest Classifier
    # ---------------------------------------------------------
    print("Training Random Forest Classifier...")
    # Initialize the Random Forest model with a fixed random state for reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    # Display clear console output indicating successful training
    print("Model trained successfully.\n")
    
    # ---------------------------------------------------------
    # Step 5: Evaluate the model
    # ---------------------------------------------------------
    print("--- Model Evaluation ---")
    
    # Calculate training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Calculate testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Display Training and Testing accuracy
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%\n")
    
    # ---------------------------------------------------------
    # Step 6: Save the trained model
    # ---------------------------------------------------------
    # Ensure the models directory exists inside PhoenixAI/
    # We use __file__ to ensure the path is relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Define the save path for the model
    model_path = os.path.join(models_dir, "model_v1.pkl")
    
    # Save the trained model using joblib
    joblib.dump(model, model_path)
    
    # Display clear console output indicating successful saving
    print("Model saved successfully.")
    print(f"Model saved to: {os.path.relpath(model_path, script_dir)}")

if __name__ == "__main__":
    main()

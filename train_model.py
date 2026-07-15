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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json

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
    # Step 4: Create an Automated Preprocessing Pipeline and Train
    # ---------------------------------------------------------
    print("Training Primary Model (Random Forest)...")
    primary_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    primary_model.fit(X_train, y_train)
    
    print("Training Backup Model (Logistic Regression)...")
    backup_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=200))
    ])
    backup_model.fit(X_train, y_train)
    
    print("Models trained successfully.\n")
    
    # ---------------------------------------------------------
    # Step 5: Evaluate the models
    # ---------------------------------------------------------
    print("--- Primary Model Evaluation ---")
    p_train_acc = accuracy_score(y_train, primary_model.predict(X_train))
    p_test_acc = accuracy_score(y_test, primary_model.predict(X_test))
    print(f"Training accuracy: {p_train_acc * 100:.2f}%")
    print(f"Testing accuracy: {p_test_acc * 100:.2f}%\n")

    print("--- Backup Model Evaluation ---")
    b_train_acc = accuracy_score(y_train, backup_model.predict(X_train))
    b_test_acc = accuracy_score(y_test, backup_model.predict(X_test))
    print(f"Training accuracy: {b_train_acc * 100:.2f}%")
    print(f"Testing accuracy: {b_test_acc * 100:.2f}%\n")
    
    # ---------------------------------------------------------
    # Step 6: Save the trained model
    # ---------------------------------------------------------
    # Ensure the models directory exists inside PhoenixAI/
    # We use __file__ to ensure the path is relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Define the save paths
    primary_path = os.path.join(models_dir, "model_primary.pkl")
    backup_path = os.path.join(models_dir, "model_backup.pkl")
    
    # Save the trained models using joblib
    joblib.dump(primary_model, primary_path)
    joblib.dump(backup_model, backup_path)
    print("Models saved successfully.")
    
    # Initialize Registry
    registry_path = os.path.join(script_dir, "model_registry.json")
    registry_data = {
        "active_model": "primary",
        "models": {
            "primary": os.path.relpath(primary_path, script_dir),
            "backup": os.path.relpath(backup_path, script_dir)
        }
    }
    with open(registry_path, "w") as f:
        json.dump(registry_data, f, indent=4)
        
    print(f"Registry initialized at: {os.path.relpath(registry_path, script_dir)}")

if __name__ == "__main__":
    main()

import os
import json
import joblib
from sklearn.metrics import accuracy_score
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(SCRIPT_DIR, "model_registry.json")
THRESHOLD = 0.90  # Accuracy threshold for failover

def get_active_model_info():
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    active_key = registry["active_model"]
    model_rel_path = registry["models"][active_key]
    model_abs_path = os.path.join(SCRIPT_DIR, model_rel_path)
    return active_key, model_abs_path, registry

def get_active_model():
    """Returns the loaded active model based on the registry."""
    _, model_path, _ = get_active_model_info()
    return joblib.load(model_path)

def switch_to_backup(registry):
    print("Switching to backup model...")
    registry["active_model"] = "backup"
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
    print("Model switched to backup successfully.")

def evaluate_and_monitor(X_test, y_test):
    """
    Evaluates the currently active model. If its accuracy falls below the threshold,
    it automatically switches to the backup model.
    """
    active_key, model_path, registry = get_active_model_info()
    print(f"Monitoring: Active model is '{active_key}'")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Current model '{active_key}' accuracy: {acc * 100:.2f}% (Threshold: {THRESHOLD * 100:.2f}%)")
    
    if acc < THRESHOLD:
        print(f"ALERT: Accuracy {acc * 100:.2f}% is below threshold {THRESHOLD * 100:.2f}%!")
        if active_key == "primary":
            switch_to_backup(registry)
        else:
            print("Already using backup model. No further fallback available.")
    else:
        print("Model is healthy. No action needed.")

if __name__ == "__main__":
    # For testing purposes, we'll load the iris dataset and artificially 
    # introduce a failure to trigger the failover.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Reset the active model to primary at the start of each run to ensure consistency
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    registry["active_model"] = "primary"
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
        
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    print("--- Normal Evaluation ---")
    evaluate_and_monitor(X_test, y_test)
    
    print("\n--- Simulating Data Drift/Accuracy Loss ---")
    # Only simulate drift if the primary model is active. 
    # If the system has already switched to backup, evaluate it on clean data.
    active_key, _, _ = get_active_model_info()
    if active_key == "primary":
        np.random.seed(42)
        y_test_simulated_bad = np.random.permutation(y_test)
        evaluate_and_monitor(X_test, y_test_simulated_bad)
    else:
        print("Backup model is active. Evaluating on clean data (no drift simulation applied).")
        evaluate_and_monitor(X_test, y_test)


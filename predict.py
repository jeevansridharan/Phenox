import numpy as np
from model_manager import get_active_model, get_active_model_info

def main():
    print("--- Model Prediction Service ---")
    
    # Get the active model safely from the manager
    active_key, _, _ = get_active_model_info()
    model = get_active_model()
    
    print(f"Using active model: '{active_key}'")
    
    # Example raw new data (sepal length, sepal width, petal length, petal width)
    new_data = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Expected: Setosa (Class 0)
        [6.0, 3.0, 4.8, 1.8]   # Expected: Virginica (Class 2)
    ])
    
    print(f"Making predictions on {len(new_data)} new samples...")
    
    # We do NOT need to scale it manually, the pipeline handles it!
    predictions = model.predict(new_data)
    
    class_names = ["setosa", "versicolor", "virginica"]
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: Predicted class {pred} ({class_names[pred]})")

if __name__ == "__main__":
    main()

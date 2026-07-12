import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = [iris.target_names[i] for i in iris.target]

# Save to CSV
df.to_csv('iris_dataset.csv', index=False)
print("Dataset successfully exported to iris_dataset.csv")

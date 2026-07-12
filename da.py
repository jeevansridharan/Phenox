from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("\nDataset Sample:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nFeatures:", iris.feature_names)
print("\nClasses:", iris.target_names)
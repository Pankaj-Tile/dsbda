# Data Visualization III
# Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
# https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
# 3. Create a boxplot for each feature in the dataset.
# 4. Compare distributions and identify outliers.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, header=None, names=column_names)

feature_types = iris.dtypes
print("Features and their types:")
print(feature_types)

plt.figure(figsize=(12, 8))
iris.drop("class", axis=1).hist(edgecolor="black", linewidth=1.2, bins=20, figsize=(12, 8))
plt.suptitle("Histograms for Each Feature", y=0.92)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=iris.drop("class", axis=1), palette="Set2")
plt.title("Box Plots for Each Feature")
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x="class", y="sepal_length", data=iris, hue="class", palette="Set2", legend=False)
plt.title("Box Plot for Sepal Length by Class")
plt.show()

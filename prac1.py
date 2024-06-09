
# Data Wrangling, I
# Perform the following operations using Python on any open source dataset (e.g., data.csv)
# 1. Import all the required Python Libraries.
# 2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear
# description of the data and its source (i.e., URL of the web site).
# 3. Load the Dataset into pandas dataframe.
# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
# function to get some initial statistics. Provide variable descriptions. Types of variables etc.
# Check the dimensions of the data frame.
# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking
# the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the
# data set. If variables are not in the correct data type, apply proper type conversions.
# 6. Turn categorical variables into quantitative variables in Python.
# In addition to the codes and outputs, explain every operation that you do in the above steps and
# explain everything that you do to import/read/scrape the data set.

import pandas as pd

url = "iris.csv"

column_names = ['sepal_lenght','sepal_width','petal_lenght','petal_width','class']

iris_df = pd.read_csv(url,names=column_names)

print("first few rows of the Iris Dataset")
print(iris_df.head())

print("\n Information about the dataset")
print(iris_df.info())

print("\n Descriptive statistics of the dataset")
print(iris_df.describe())

print("\nDimensions of the dataset (rows,columns) : ",iris_df.shape)

print("\nData type pf Variables : ")
print(iris_df.dtypes)

iris_df = pd.get_dummies(iris_df, columns=['class'], drop_first=True)

print("\nUpdated DataFrame after one-hot encoding:")
print(iris_df.head())
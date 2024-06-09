# Descriptive Statistics - Measures of Central Tendency and variability
# Perform the following operations on any open source dataset (e.g., data.csv)
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for
# a dataset (age, income etc.) with numeric variables grouped by one of the qualitative
# (categorical) variable. For example, if your categorical variable is age groups and
# quantitative variable is income, then provide summary statistics of income grouped by the
# age groups. Create a list that contains a numeric value for each response to the categorical
# variable.
# 2. Write a Python program to display some basic statistical details like percentile, mean,
# standard deviation etc. of the species of 'Iris-setosa','Iris-versicolor' and 'Iris-versicolor'
# of iris.csv dataset.
# Provide the codes with outputs and explain everything that you do in this step.


import pandas as pd


titanic_df = pd.read_csv('titanic.csv')

print("First few rows of the Titanic dataset:")
print(titanic_df.head())


grouped_stats = titanic_df.groupby('Pclass')['Age'].describe()

print("\nSummary statistics of Age grouped by Pclass:")
print(grouped_stats)

mean_age_by_class = titanic_df.groupby('Pclass')['Age'].mean().tolist()

print("\nMean Age for each Passenger Class:")
print(mean_age_by_class)


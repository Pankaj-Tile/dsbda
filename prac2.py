# Data Wrangling II
# Create an "Academic performance" dataset of students and perform the following operations using
# Python.
# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
# inconsistencies, use any of the suitable techniques to deal with them.
# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable
# techniques to deal with them.
# 3. Apply data transformations on at least one of the variables. The purpose of this
# transformation should be one of the following reasons: to change the scale for better
# understanding of the variable, to convert a non-linear relation into a linear one, or to
# decrease the skewness and convert the distribution into a normal distribution.
# Reason and document your approach properly.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

data = {
    'Student_ID': range(1, 101),
    'Math_Score': np.random.randint(50, 100, size=100),
    'English_Score': np.random.randint(40, 95, size=100),
    'Science_Score': np.random.randint(55, 98, size=100),
    'Attendance_Percentage': np.random.uniform(70, 100, size=100),
    'Study_Hours_Per_Day': np.random.uniform(1, 6, size=100),
}

academic_df = pd.DataFrame(data)

academic_df.loc[10:20, 'Math_Score'] = np.nan
academic_df.loc[30:40, 'English_Score'] = np.nan
academic_df.loc[50:60, 'Science_Score'] = np.nan
academic_df.loc[70:80, 'Attendance_Percentage'] = np.nan

print("First few rows of the Academic Performance dataset:")
print(academic_df.head())

academic_df.fillna(academic_df.mean(), inplace=True)
academic_df[academic_df < 0] = np.nan

print("\nUpdated dataset after handling missing values and inconsistencies:")
print(academic_df.head())


numeric_vars = ['Math_Score', 'English_Score', 'Science_Score', 'Attendance_Percentage', 'Study_Hours_Per_Day']

z_scores = (academic_df[numeric_vars] - academic_df[numeric_vars].mean()) / academic_df[numeric_vars].std()
outliers = (z_scores > 3) | (z_scores < -3)

academic_df[outliers] = np.nan

print("\nDataset after handling outliers:")
print(academic_df.head())

academic_df['Log_Study_Hours'] = np.log1p(academic_df['Study_Hours_Per_Day'])

print("\nDataset after log transformation:")
print(academic_df.head())


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(academic_df['Study_Hours_Per_Day'], kde=True)
plt.title('Study_Hours_Per_Day Distribution')

plt.subplot(1, 2, 2)
sns.histplot(academic_df['Log_Study_Hours'], kde=True)
plt.title('Log_Study_Hours Distribution')

plt.show()


# Data Analytics I
# Create a Linear Regression Model using Python/R to predict home prices using Boston Housing
# Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains
# information about various houses in Boston through different parameters. There are 506 samples
# and 14 feature variables in this dataset.
# The objective is to predict the value of prices of the house using the given features.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

boston_data = pd.read_csv('train.csv')

print(boston_data.head())

X = boston_data.drop(['ID', 'indus'], axis=1)
y = boston_data['indus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

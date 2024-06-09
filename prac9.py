# Data Visualization II
# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution
# of age with respect to each gender along with the information about whether they survived
# or not. (Column names : 'sex' and 'age')
# 2. Write observations on the inference from the above statistics.


import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

plt.figure(figsize=(12, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')
plt.title('Distribution of Age by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

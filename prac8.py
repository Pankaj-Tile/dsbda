# Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information
# about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to
# see if we can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
# is distributed by plotting a histogram


import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.countplot(x='class', hue='survived', data=titanic, palette='Set1')
plt.title('Survival Count by Passenger Class')


plt.figure(figsize=(12, 6))

sns.histplot(titanic['fare'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Ticket Prices (Fare)')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
plt.show()


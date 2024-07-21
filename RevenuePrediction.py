import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure inline plotting for Jupyter notebooks

# Importing the dataset
IceCream = pd.read_csv("IceCreamData.csv")

# Displaying the first and last few rows of the dataset
print(IceCream.head())
print(IceCream.tail())

# Displaying statistical summary and info about the dataset
print(IceCream.describe())
print(IceCream.info())

# Visualizing the dataset using seaborn
sns.jointplot(x='Temperature', y='Revenue', data=IceCream)
sns.pairplot(IceCream)
sns.lmplot(x='Temperature', y='Revenue', data=IceCream)

# Defining the dependent and independent variables
y = IceCream['Revenue']
X = IceCream[['Temperature']]

from sklearn.model_selection import train_test_split

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("X_train shape:", X_train.shape)

from sklearn.linear_model import LinearRegression

# Instantiating the LinearRegression object
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Intercept (b): ', regressor.intercept_)

# Predicting the test set results
y_predict = regressor.predict(X_test)
print("Predictions: ", y_predict)
print("Actual values: ", y_test.values)

# Visualizing the training set results
plt.figure()
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Temperature [degC]')
plt.ylabel('Revenue [dollars]')
plt.title('Revenue Generated vs. Temperature @ Ice Cream Stand (Training dataset)')
plt.show()

# Visualizing the test set results
plt.figure()
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.xlabel('Temperature [degC]')
plt.ylabel('Revenue [dollars]')
plt.title('Revenue Generated vs. Temperature @ Ice Cream Stand (Test dataset)')
plt.show()

# Predicting the revenue at a specific temperature (e.g., 30 degrees)
temp_to_predict = np.array([[30]])
y_predict_30 = regressor.predict(temp_to_predict)
print("Predicted revenue at 30°C: ", y_predict_30[0])

#Polynomial Regression
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
path = 'Position_Salaries.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the Regression Model to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

#Prediction a new result with Polynomial Regression
regressor.predict(poly_reg.fit_transform(6.5))

#Visualizing the Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
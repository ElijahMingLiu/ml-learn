#Support Vector Regression 支持向量回归

太简单，就不具体解释了。

---

---

全部代码:

```python
#SVR
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape((len(y),1)))

#Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf',)
regressor.fit(X, y)

#Prediction a new result with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Visualizing the SVR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```



代码github地址：[svm.py](../resources/svm.py)


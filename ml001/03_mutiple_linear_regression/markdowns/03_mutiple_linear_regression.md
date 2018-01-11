#multiple linear regression 多元线性回归





###最开始，进行数据预处理

首先，导入数据集：

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

数据集如下图，共50个数据项，特征分别为：

R&D Spend(研发花费) Administration(管理经费) Marketing Spend(市场花费) 

要预测的内容为 Profit(盈利)

![dataset_screenshot](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\dataset_screenshot.jpg)

然后，建造分类变量的dummy variable：

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
```

为了防止dummy variable trap(虚拟变量陷阱)所产生的Multicollinearity(多重共线性)，需要将其避免，具体方法就是把每组dummy variable的其中一列移除：

```python
X = X[:,1:]
```

将数据集分为训练集与测试集：

```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```



### 下面是基础的多元回归模型的训练过程

首先，用多线性回归器对训练集进行拟合，并用在测试集上进行训练：

```python
#Fitting Multiple Linear Regression to the  Training set
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
```

由于在上一次的简单线性回归中给出了图片示例，在此就不进行展示了。



### 下面是逐步回归，采用Backward Elimination(反向淘汰)

由于在回归过程中，有很多变量是不需要的(p值较高)，所以要将其淘汰，具体步骤如下：

step 1:为p值选择一个阈值SL(Significance leve)，这里为0.05

step 2:使用所有的可用的变量，训练出模型

step 3:如果p值最高的一个变量，如果其p值 P>SL，跳到step 4，否则跳到最后

step 4:将此变量删除

step 5:利用剩下的变量拟合模型



就如此，不断的循环以上5步，直到没有一个变量的p值大于SL，就停止。

以下为具体过程：

首先导入需要使用的工具库(这里使用statsmodels工具库，因为其可以查看统计数值)，并将所有变量放入数据集中，即X_opt。

其中，需要使用np.append将一列1放入数据集的第一列代表其bias，即截距。即在一列$[1, 1, 1, 1...]^T$的后面加上X

```python
import statsmodels.formula.api as sm
X = np.append(values=X,arr=np.ones((50,1)).astype(int),axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
```

建立模型，并使用所有变量拟合模型，并使用regressor_OLS.summary()查看其p值

```python
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```

结果显示如下，可看出，x2的p值为0.990，是最大的，且大于0.05(x0是bias，就是上面的const)

![be1](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\be1.jpg)

将x2删除，继续拟合模型，并查看其p值

```python
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```

结果显示如下，可看出，x1的p值为0.940，是最大的，且大于0.05

![be2](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\be2.jpg)



将x1删除，继续拟合模型

```python
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```

结果显示如下，可看出，x2的p值为0.602，是最大的，且大于0.05

![be3](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\be3.jpg)

将x2(就是原来的第4个变量)删除，继续拟合模型

结果显示如下，可看出，x2的p值为0.060，还是大于0.05

![be4](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\be4.jpg)

将x2(即原来的第5个变量)，然后继续拟合

```python
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```

结果显示如下，现在没有一个变量的p值大于0.05了，所以这就是最优的模型。

可以发现，第3个变量(R&D Spend)对Profit的影响是最重要的，而且是唯一的预测变量(在SL=0.05的条件下)。

![be5](C:\Users\Administrator\Dropbox\博客\ml-learn\ml001\03_mutiple_linear_regression\markdowns\pic\be5.jpg)

关键词

> back elimiation：反向淘汰，用于逐步回归法中淘汰对预测影响过小的变量，防止过多的无用变量对预测导致不良影响，以便防止过拟合

> dummy variable trap：虚拟变量陷阱，防止虚拟变量间产生的多重共线性（有多重共线性的数据集，使用回归分析会产生很大的问题）

> P-Value：P值，用来判定假设检验结果的一个参数，就是当原假设为真时所得到的样本观察结果或更极端结果出现的概率，如p值过小，会选择使用备择假设。在此例中，
>
> 原假设：该变量的权值为0
>
> 备择假设：该变量的权值不为0





---

---

全部代码:

```python
#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the  Training set
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(values=X,arr=np.ones((50,1)).astype(int),axis=1)
X_opt = X[:, [0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
```



代码github地址：[x](x)


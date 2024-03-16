# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ABINAV SANKAR S
RegisterNumber: 212222040002
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)
#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/7e9a0db3-e69f-464a-b763-778bb634d3e3)
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/85657f61-c270-41c6-b22f-4b26f0e41904)
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/cea603c4-a8ca-44c2-bbba-16f23bdfab47)
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/ef4c00fc-47fe-45d5-bc2c-598fe8f46a2f)
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/de66d5bb-1602-4c04-bf03-c484eb8a14c1)
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/1c2d260f-9914-494e-afe6-542b2fab4ae4)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

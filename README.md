# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

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
## df.head()
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/9ae71d6f-050e-4dbb-885e-bb18b2e57509)
## df.tail()
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/4075d985-5d8c-424a-af2e-5592df0f4dbd)
## Array Value of x
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/f19e4e6d-c1be-46e2-b909-c5a4835e60f8)
## Array Value of Y
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/89537c67-c6ce-458c-82cb-3119d041b413)
## Values of Y prediction 
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/f1ff306d-389c-4dc2-97b1-aee1ce687e18)
## Array Values of Y test
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/6e79c934-5729-4081-8bb3-f6d5debe8b12)
## Training Set graph 
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/2f25c025-d6c5-499c-8915-ce0710afdf59)
## Test Set graph
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/17e60137-6dcb-4ecb-9f1a-1a63edb60e7a)
## Values of MSE,MAE & RMSE
![image](https://github.com/Abinavsankar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119103734/592af942-bda4-4cb4-a70a-2eb48aef9b21)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

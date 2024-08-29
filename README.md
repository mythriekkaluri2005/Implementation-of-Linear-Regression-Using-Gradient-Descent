# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1:Start

Step-2:Intialize weights randomly.

Step-3:Compute predicted.

Step-4:Compute gradient of loss function.

Step-5:Update weights using gradient descent.

Step-6:End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: E.Mythri
RegisterNumber: 212223240034
*/


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta usinng gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Desktop/50_Startups.csv",header=None)
data.head()
X=(data.iloc[1:,:-2].values) 
X1=X.astype(float)

scaler=StandardScaler()

y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)

Y1_Scaled=scaler.fit_transform(y)

print(X)

print(X1_Scaled)
```

## Output:
![Screenshot 2024-08-29 171800](https://github.com/user-attachments/assets/de3a1564-3fbc-422e-b19e-a11b9b2da036)

![Screenshot 2024-08-29 171947](https://github.com/user-attachments/assets/5f10d1e0-1391-444a-82ec-82142ffe92ae)

![Screenshot 2024-08-29 172327](https://github.com/user-attachments/assets/005e6adc-f19f-435b-80bd-b611b5ded98c)

![Screenshot 2024-08-29 172041](https://github.com/user-attachments/assets/a80ff115-6575-403a-9281-2b1ba6442a4b)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

# Implementation-of-Linear-Regression-Using-Gradient-Descent
### Date:
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1: Start the Program

STEP 2: Import the required library and read the dataframe.

STEP 3: Write a function computeCost to generate the cost function.

STEP 4: Perform iterations og gradient steps with learning rate.

STEP 5: Plot the Cost function using Gradient Descent and generate the required graph

STEP 6: Stop the program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RAMYA R
RegisterNumber:  212223230169
*/



import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        
        #update theta using gradient descent 
        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Documents/New folder (2)/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled, Y1_Scaled)
#predict target value for a new data point
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")



```

## Output:
![image](https://github.com/user-attachments/assets/0b9d5cff-56e2-4820-8b35-cc1753389ba1)

![image](https://github.com/user-attachments/assets/68888f04-b12f-4dcc-9d3d-79b29a3875f1)

![image](https://github.com/user-attachments/assets/06fc24af-b581-454e-895f-34bbd5c1516e)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.


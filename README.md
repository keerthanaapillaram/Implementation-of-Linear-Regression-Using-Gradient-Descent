# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Load necessary libraries for data handling, metrics, and visualization.
2. Load Data: Read the dataset using pd.read_csv() and display basic information.
3. Initialize Parameters: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. Gradient Descent: Perform iterations to update m and c using gradient descent.

5. Plot Error: Visualize the error over iterations to monitor convergence of the model.


## Program and Outputs:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: P Keerthana
RegisterNumber: 212223240069
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
```

```
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
```

```
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
```

![image](https://github.com/user-attachments/assets/6be4d690-e715-4967-9d83-17ce6aa68722)

```
X=(data.iloc[1:, :-2].values)
print(X)

```


![image](https://github.com/user-attachments/assets/ce29cb0d-6c62-4ca7-8728-8e7e5dacf5b2)

```
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```


![image](https://github.com/user-attachments/assets/c26d91a9-91bf-475a-990f-d63770c9195f)

```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
```
```
print(X1_Scaled)
```

![image](https://github.com/user-attachments/assets/d4598e9b-6731-4201-ae54-10bd1713a101)

```
print(Y1_Scaled)
```

![image](https://github.com/user-attachments/assets/9f042ffb-1f11-45e8-aece-a08f27d55f31)

```
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```


![image](https://github.com/user-attachments/assets/1e7d4e77-eac2-4824-9c6f-b8da99925aa8)

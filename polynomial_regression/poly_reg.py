# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:54:00 2023

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Splitting the dataset into the training set and test set 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)

#Feature Scaling 
sc_x=StandardScaler()
x_train=sc_x.fit(x_train)
y_train=sc_x.fit(x_test)
#fitting the poly regression to the dataset
poly_reg=PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly, y)


#visualising the polynmial regression
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(x_poly),color='blue')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('salary_data_lyst3360 (1).csv')
#print(data)

#x=data['YearsExperience'].values
x=data.iloc[::-1].values
#print(x)
#y=data['Salary'].values
y=data.iloc[:,1].values
#print(y)

#plt.scatter(x,y)
#plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)
#print(x_train)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
z=3
#y_pred=regressor.predict(x_test)
#print(y_pred)
#yy=pd.DataFrame(y_pred,y_test)
#print(yy)
#y_pred1=regressor.predict([[7]])
#print(y_pred1)

#coefficient of intercept
print(regressor.intercept_)

#coeffient of coefficirnt
print(regressor.coef_)

plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train))
plot.show()


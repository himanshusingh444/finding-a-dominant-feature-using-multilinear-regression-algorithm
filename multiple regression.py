
# multiple regressions
 
#importing libraries
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv') 
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
#changing into dummy variable

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
 
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding dummy trap
x = x[:,1:]


#spliting into tarin ansd test sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#fitting regressions
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
 #prediction
y_pred = regressor.predict(x_test)


# backward elimination method to find optimal value


#building a optimal model using a backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x  ,axis = 1) #adding a column which will define b0 
x_opt = x[:,[0,1,2,3,4,5]] #optimal of x means accurate by which we eill find more accurate affecting on dependent value 
#fit the model into regression
regressor_OLS= sm.OLS(endog = y ,exog = x_opt).fit()
#using summary function we can get pvalue and a statical array
regressor_OLS.summary()
#remoing x2 because it is too high from sl value

x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

#now remving x1 it has to high value in cmparison to sl value
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()
 #now removing x2
x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
#REMOVING 5
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

# COMPANY SHOULD FOCUS  MARKETTING SPEND
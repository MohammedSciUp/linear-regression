#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with SKlearn 

# theory is found in the PDF :https://github.com/MohammedSciUp/Mathematics-for-linear-regression-/blob/master/mathematics%20for%20linear%20regression.pdf

# In[127]:


#--------------importing libraries

import pandas as pd
df = pd.read_excel('D:\\1\\co2.xlsx')
df.head()


# In[128]:


co2_engsize=df[['ENGINESIZE','CO2EMISSIONS']]
co2_engsize


# In[110]:


co2_engsize.describe()


# In[129]:


import matplotlib.pyplot as plt 
ENGINE_SIZE = co2_engsize.sort_values('ENGINESIZE',ascending=False) 
CO2_EMISSIONS = co2_engsize.sort_values('CO2EMISSIONS',ascending=False) 
plt.scatter(ENGINE_SIZE,CO2_EMISSIONS,color='#cc95e6')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2Emission')
plt.show()


# In[130]:


viz = co2_engsize[['ENGINESIZE','CO2EMISSIONS']]
viz.hist(color='#edb418')
#plt.grid()
plt.show()


# In[131]:


# sampling our data with shuffling ... test set 20% , training set 80% of the overall data. 
msk = np.random.rand(len(df)) < 0.8
train = co2_engsize[msk]
test = co2_engsize[~msk]
df[msk]


# In[132]:


#---- scatter plot training data

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='#aae6e4')
plt.xlabel('ENGINESIZE')
plt.ylabel("CO2EMISSION")
plt.show()


# In[136]:


#---- LINEAR REGRESSION MODEL 

import numpy as np
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
print ('theta 0: ', regr.coef_)
print ('theta 1: ',regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='#89e386')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '#6127c4')   #the line equation
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.grid()
plt.show()

  #### --- there was issue whith test set no infinite NaN values .. fixed with deleting all non used columns rows in excel in can deleting or ignore by selecting size of column


# In[139]:


# test set ------------------------------------**

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("M A error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)) , "sum squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2) , "R2-score: %.2f" % r2_score(test_y_hat , test_y))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





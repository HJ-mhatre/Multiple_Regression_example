#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression with Dummies - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year_view.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. 
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size', 'year', and 'view'.
# 
# #### Regarding the 'view' variable:
# There are two options: 'Sea view' and 'No sea view'. You are expected to create a dummy variable for view and include it in the regression
# 
# Good luck!

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as st

sb.set()


# ## Load the data

# In[2]:


data1=pd.read_csv('real_estate_price_size_year_view.csv')


# In[5]:


data1


# In[6]:


data1.describe()




y=data1['price']
x1=data1[['size', 'year']]


# ## Create a dummy variable for 'view'

# In[16]:


data1

data1.describe()
data1.summary()


# In[11]:


data2=data1.copy()

data2['view']=data2['view'].map({'No sea view':0, 'Sea view':1})

data2
data2.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[13]:


y=data2['price']
x1=data2[['size','view']]


# ### Regression

# In[15]:


x=st.add_constant(x1)
results=st.OLS(y,x).fit()
results.summary()


# In[24]:


plt.scatter(data2['size'],y)
yhat_seaview=7.748e+04+218.7521*data2['size']+5.756e+04
yhat_noseaview=7.748e+04+218.7521*data2['size']
fig=plt.plot(data2['size'], yhat_seaview, lw=2, c='orange')
fig=plt.plot(data2['size'], yhat_noseaview, lw=2, c='red')
plt.xlabel('size',fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()






# In[ ]:


#houses having sea view is more costlier than those who does not have sea view


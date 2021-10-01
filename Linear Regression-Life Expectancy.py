#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn import linear_model
dataframe=pd.read_csv('Life Expectancy Data.csv')
dataframe.head()


# In[3]:


dataframe.info()


# In[4]:


dataframe.isnull().any()


# In[5]:


dataframe['Life expectancy '].fillna(dataframe['Life expectancy '].median(),inplace=True)
dataframe['Adult Mortality'].fillna(dataframe['Adult Mortality'].median(),inplace=True)
dataframe['Alcohol'].fillna(dataframe['Alcohol'].median(),inplace=True)
dataframe['Hepatitis B'].fillna(dataframe['Hepatitis B'].median(),inplace=True)
dataframe[' BMI '].fillna(dataframe[' BMI '].median(),inplace=True)
dataframe['Polio'].fillna(dataframe['Polio'].median(),inplace=True)
dataframe['Total expenditure'].fillna(dataframe['Total expenditure'].median(),inplace=True)
dataframe['Diphtheria '].fillna(dataframe['Diphtheria '].median(),inplace=True)
dataframe['GDP'].fillna(dataframe['GDP'].median(),inplace=True)
dataframe['Population'].fillna(dataframe['Population'].median(),inplace=True)
dataframe[' thinness  1-19 years'].fillna(dataframe[' thinness  1-19 years'].median(),inplace=True)
dataframe['Income composition of resources'].fillna(dataframe['Income composition of resources'].median(),inplace=True)
dataframe['Schooling'].fillna(dataframe['Schooling'].median(),inplace=True)
dataframe[' thinness 5-9 years'].fillna(dataframe[' thinness 5-9 years'].median(),inplace=True)


# In[6]:


dataframe.isnull().any()


# In[7]:


dataframe.head()


# In[8]:


dataframe.drop(['Country','Status'],axis=1,inplace=True)


# In[9]:


plt.figure(figsize=(10,10))
plt.xlabel('Adult Mortaity')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe['Adult Mortality'], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe['Adult Mortality'], dataframe['Life expectancy '],1)
plt.plot(dataframe['Adult Mortality'],m*dataframe['Adult Mortality']+b)


# In[10]:


plt.figure(figsize=(10,10))
plt.xlabel('Infant Deaths')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe['infant deaths'], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe['infant deaths'], dataframe['Life expectancy '],1)
plt.plot(dataframe['infant deaths'], m*dataframe['infant deaths']+b)


# In[11]:


plt.figure(figsize=(10,10))
plt.xlabel('Schooling')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe['Schooling'], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe['Schooling'], dataframe['Life expectancy '],1)
plt.plot(dataframe['Schooling'], m*dataframe['Schooling']+b)


# In[12]:


plt.figure(figsize=(10,10))
plt.xlabel('Polio')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe['Polio'], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe['Polio'], dataframe['Life expectancy '],1)
plt.plot(dataframe['Polio'], m*dataframe['Polio']+b)


# In[13]:


plt.figure(figsize=(10,10))
plt.xlabel('under-five deaths')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe['under-five deaths '], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe['under-five deaths '], dataframe['Life expectancy '],1)
plt.plot(dataframe['under-five deaths '], m*dataframe['under-five deaths ']+b)


# In[14]:


plt.figure(figsize=(10,10))
plt.xlabel('thinness 1-19 years')
plt.ylabel('Life expectancy ')
plt.scatter(dataframe[' thinness  1-19 years'], dataframe['Life expectancy '])
m,b=np.polyfit(dataframe[' thinness  1-19 years'], dataframe['Life expectancy '],1)
plt.plot(dataframe[' thinness  1-19 years'], m*dataframe[' thinness  1-19 years']+b)


# In[15]:


y=dataframe['Life expectancy ']
dataframe.info()


# In[16]:


x=dataframe


# In[17]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# In[18]:


from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(xtrain,ytrain)


# In[19]:


lreg.score(xtrain,ytrain)


# In[20]:


lreg.score(xtest,ytest)


# In[21]:


print(lreg.intercept_)
print(lreg.coef_)


# In[22]:


ypred=lreg.predict(xtest)


# In[26]:


from sklearn import metrics
print("Mean Absolute error is: ",metrics.mean_absolute_error(ytest,ypred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv(r'E:\Udemy corurse\[DesireCourse.Net] Udemy - Machine Learning A-Z™ Hands-On Python & R In Data Science\1.Machine Learning A-Z New\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv')


# In[3]:


dataset.head()


# In[4]:


data = dataset.values
X = data[:, :-1]
y = data[:, 1]


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


lm =LinearRegression()


# In[9]:


lm.fit(X_train,y_train)


# In[10]:


y_prediction = lm.predict(X_test)


# In[11]:


plt.scatter(X_train ,y_train, color= 'red')
plt.plot(X_train, lm.predict(X_train),color='blue')


# In[12]:


plt.scatter(X_test ,y_test, color= 'red')
plt.plot(X_train, lm.predict(X_train),color='blue')


# In[ ]:





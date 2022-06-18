#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data=pd.read_csv("data_2_var.csv")


# In[3]:


data.head()


# In[4]:


print(data.isnull().sum())


# In[9]:


import seaborn as sns


# In[10]:


sns.scatterplot(x="x",y="y",data=data)


# In[14]:


sns.boxplot(data["y"])


# In[15]:


Q1 = np.percentile(data["y"], 25,
                   interpolation = 'midpoint')
Q3 = np.percentile(data["y"], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1 
print("Old Shape: ", data.shape)
upper = np.where(data["y"] >= (Q3+1.5*IQR))
lower = np.where(data["y"] <= (Q1-1.5*IQR)) 
data.drop(upper[0], inplace = True)
data.drop(lower[0], inplace = True)

print("New Shape: ", data.shape)


# In[16]:


sns.scatterplot(x="x",y="y",data=data)


# In[18]:


print(IQR)


# In[19]:


x = np.array(data["x"])
y = np.array(data["y"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[20]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[21]:


features = np.array([[230.1, 37.8,]])
print(model.predict(features))


# In[ ]:





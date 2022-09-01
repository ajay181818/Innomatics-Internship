#!/usr/bin/env python
# coding: utf-8

# # Task - Predict the diamond price

# 
# import library

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


# loading the dataset
# 

# In[ ]:


df=pd.read_csv("diamonds.csv")


# In[ ]:


df.head()


# Step - 2: Perform the EDA on the given dataset

# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# we did not have any null value in our dataset

# In[7]:


sn.boxplot(x="x",data=df)


# In[8]:


sn.boxplot(x="y",data=df)


# In[9]:


sn.boxplot(x="z",data=df)


# In x,y,z features ther are some outliers and also some data points are at zero so we have to remove them 

# In[10]:


df=df.drop(df[df['x']==0].index)
df=df.drop(df[df['y']==0].index)
df=df.drop(df[df['z']==0].index)


# In[11]:


df.shape


# In[12]:


df = df[(df["x"]<30)]
df = df[(df["y"]<30)]
df = df[(df["z"]<30)&(df["z"]>2)]


# In[ ]:


sn.pairplot(df,hue="cut")


# In[13]:


sn.boxplot(x="z",data=df)


# we have remove all the outliers from x y z columns

# In[14]:


sn.barplot(x="cut",y="price",data=df)


# It is clear that premium cut diamond is most expensive and ideal cut is most cheaper in price

# In[15]:


sn.barplot(x="color",y="price",data=df)


# It is clear that j color diamond is very expensive compared to E color

# In[16]:


sn.barplot(x="clarity",y="price",data=df)


# We can sea s12 clarity diamond are most expensive

# In[17]:


sn.histplot(x="table",data=df,kde=True)


# In[18]:


sn.boxplot(x="table",data=df)


# In[19]:


q1 = df['table'].quantile(0.25)
q3 = df['table'].quantile(0.75)
print(q1,q3)


# In[20]:


iqr=q3-q1
iqr


# In[21]:


upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr
print(upper_limit,lower_limit)


# In[22]:


df = df[(df["table"]<63.5)&(df["table"]>51.5)]


# In[23]:


df.shape


# In[24]:


sn.boxplot(x="table",data=df)


# we remove outliers from table coulms also

# In[25]:


sn.boxplot(x="depth",data=df)


# In[26]:


q1_d = df['depth'].quantile(0.25)
q3_d = df['depth'].quantile(0.75)
print(q1_d,q3_d)
iqr_d=q3_d-q1_d
upper = q3_d+ 1.5 * iqr_d
lower = q1_d - 1.5 * iqr_d
print(upper,lower)


# In[27]:


df = df[(df["depth"]<64.6)&(df["depth"]>59)]


# In[28]:


sn.boxplot(x="depth",data=df)


# In[29]:


sn.boxplot(x="carat",data=df)


# Step - 3: Handle Categorical Columns

# In[30]:


from sklearn import preprocessing


# In[31]:


label_encoder = preprocessing.LabelEncoder()
df['color']= label_encoder.fit_transform(df['color'])
df['cut']= label_encoder.fit_transform(df['cut'])
df['clarity']= label_encoder.fit_transform(df['clarity'])
  


# In[32]:


df


# In[33]:


correrlation= df.corr()
sn.heatmap(correrlation,annot=True)


# we can see that price has maximum correlation with carat,x,y,z and the leat correlation is with clarity,depta

# Step - 4: Normalize the data
# 

# In[34]:


from sklearn.preprocessing import StandardScaler
standardized_df = StandardScaler().fit_transform(df)
print(standardized_df.shape)


# In[35]:


df


# Step - 5: Split the data - Test and Train

# In[36]:


x=df[['carat','cut','color','clarity','depth','table','x', 'y','z']]


# In[37]:


y=df['price']


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# Building machine learning model

# knn model

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.predict(X_test))


# accuracy of model

# In[45]:


print(knn.score(X_test, y_test))


# In[46]:


from sklearn.linear_model import LinearRegression

linear_model=LinearRegression()


# Linear regression model

# In[47]:


linear_model.fit(X_train,y_train)


# In[48]:


y_pred =print(linear_model.predict(X_test))


# In[50]:


linear_model.coef_


# In[52]:


cofficent=pd.DataFrame(linear_model.coef_,x.columns,columns=['Coeff'])
cofficent


# accuracy of model

# In[53]:


print(linear_model.score(X_test, y_test))


# In[54]:


from sklearn import metrics


# In[68]:


result=linear_model.predict(X_test)
print("Mean absolute error is ",metrics.mean_absolute_error(y_test,predictions))


# In[70]:


print("Mean Squared error is ",metrics.mean_squared_error(y_test,predictions))


# In[73]:


print("Root-mean-square error is ",np.sqrt(metrics.mean_squared_error(y_test,result)))


# Random forest model

# In[74]:


from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor()
RF.fit(X_train,y_train)


# In[78]:


y_predictions =print(RF.predict(X_test))


# In[79]:


predictionsRF=RF.predict(X_test)


# In[81]:


r2=metrics.explained_variance_score(y_test,predictionsRF)
r2


# In[ ]:





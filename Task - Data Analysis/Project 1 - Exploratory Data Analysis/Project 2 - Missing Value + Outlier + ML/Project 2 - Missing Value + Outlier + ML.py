#!/usr/bin/env python
# coding: utf-8

# # EDA + Missing values and Outliers - Detection and Treatment + Model Building and MORE 

# # Step - 1 - Introduction -> Give a detailed data description and objective
# 

# # About Dataset

# An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.
# 
# Fields The dataset contains 15 columns Target filed: Income -- The income is divide into two classes: <=50K and >50K Number of attributes: 15 -- These are the demographics and other features to describe a person
# 
# We can explore the possibility in predicting income level based on the individual’s personal information.
# 
# 

# # Objective

# Objective of this product is to perform EDA, find the missing values if any, find the outliers, and lastly build various Machine Learning models considering ‘income’ as target variable and compare the performance of each of the ML Model.

# # Step - 2 - Import the data and perform basic pandas operations 
# 

# importing all the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# importing the dataset

# In[2]:


df=pd.read_csv("adult.csv")


# viewing the dataset

# In[3]:


df.head()


# checking the shape of the dataset

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.columns


# checking the data types of coulmns

# In[8]:


df.dtypes


# In[9]:


df.nunique()


# checking the null values in the dataset

# In[10]:


df.isnull().sum()


# counted the duplicates rows

# In[11]:


df[df.duplicated()].count()


# droping the duplicated rows

# In[12]:


df=df[~df.duplicated()]


# In[13]:


df.shape


# # Step - 3 - Univariate Analysis -> PDF, Histograms, Boxplots, Countplots, etc..
# 

# dividing the categorical coulmn and numerical columns

# In[14]:


cat_df = df.select_dtypes(include=["object"])
num_df = df.select_dtypes(include=['int64', 'float64'])


# In[15]:


sns.countplot(x="income",data=df)


# In[16]:


cat_df


# In[17]:


plt.subplots(figsize=(25, 50))
count = 1
for i in cat_df.columns:
    plt.subplot(12, 3, count)
    sns.countplot(x = cat_df[i])
    count += 1
plt.show()


# In[18]:


plt.subplots(figsize=(25, 50))
count = 1
for i in num_df.columns:
    plt.subplot(12, 3, count)
    sns.histplot(x = num_df[i])
    count += 1
plt.show()


# # Step - 4 - Bivariate Analysis

# In[19]:


sns.catplot(
    data=df, x="race", y="hours-per-week", col="gender")


# In[20]:


sns.boxplot(data=df, x="relationship", y="age", hue="income", dodge=False)


# In[21]:


sns.violinplot(data=df, x="age", y="marital-status")


# In[22]:


sns.boxenplot(data=df, x="educational-num", y="workclass")


# In[23]:


sns.heatmap(num_df.corr(), annot=True)


# In[24]:



plt.subplots(figsize=(15, 30))
count = 1
for i in range(len(num_df.columns)-1):
    plt.subplot(14, 1, count)
    sns.scatterplot(x=num_df.columns[i], y=num_df.columns[i+1],
                        data=num_df, color='green')
    count += 1
plt.show()


# # Step - 5 - Find and treat the outliers and missing values in each column 

# In[25]:


plt.subplots(figsize=(25, 50))
count = 1
for i in num_df.columns:
    plt.subplot(12, 3, count)
    sns.boxplot(x = num_df[i])
    count += 1
plt.show()


# In[26]:


q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
print("q1",q1,"q3",q3)

iqr=q3-q1
print("iqr",iqr)

upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr
print("upper limit ",upper_limit,"lower limit",lower_limit)


# In[27]:


df = df[(df["age"]<78)&(df["age"]>-2.0)]


# In[28]:


q11 = df['educational-num'].quantile(0.25)
q33 = df['educational-num'].quantile(0.75)
print("q1",q1,"q3",q3)

iqr1=q33-q11
print("iqr",iqr)

upper_limit = q33 + 1.5 * iqr
lower_limit = q11 - 1.5 * iqr
print("upper limit ",upper_limit,"lower limit",lower_limit)


# In[29]:


df = df[(df["educational-num"]<42)&(df["age"]>-21.0)]


# In[30]:


sns.boxplot(data=df,x="age")


# In[31]:


sns.boxplot(data=df,x="educational-num")


# # Step - 6 - Apply appropriate hypothesis tests to verify the below mentioned questions
# 

# Research question 1- Is there a relationship between occupation and gender? (i.e. does the preference of occupation depend on the gender)
# 

# In[32]:


sns.countplot(y="occupation", hue="gender",data=df)


# According to the plot above, whether a person's occupation is a Machine-op-inspct or Farming-fishing or Protective-serv or Prof-specialty or Craft-repair or Exec-managerial or Tech-supportor Sales or Transport-moving or Handlers-cleaners or Armed-Forces, males outnumber females.
# 
# When a person's employment is Other-service, Adm-clerical, or Priv-house-serv, females outnumber males in these occupations. Yes, there is a connection between gender and occupation. Gender influences occupation.

# # Testing the hypothesis:

# 
# Null Hypothesis: there is no relationship between occupation and gender
# 
# ALternate Hypothesis: there is  relationship between occupation and gender

# In[33]:


cross_tab = pd.crosstab(df['occupation'],df['gender'])
cross_tab


# In[34]:


from scipy.stats import chi2_contingency

stat, p, dof, expected = chi2_contingency(cross_tab)

alpha = 0.05
print("p value = {}".format(p))
if p <= alpha:
    print("{} <= {}".format(p,alpha))
    print("there is  relationship between occupation and gender. So accept the  alternative Hypothesis")
else:
    print("there is no relationship between occupation and gender. So reject the  Null Hypothes")


# Research question 2- Is there a relationship between gender and income?

# In[35]:


sns.countplot(x="income", hue="gender",data=df)


# From the above plot we notice that income is more fo male and less for female.
# 
# So, inorder to conclude we need to make hypothesis regarding this statement

# # Testing the hypothesis:

# 
# Null Hypothesis: there is no relationship between income and gender
# 
# ALternate Hypothesis: there is  relationship between income and gender

# In[36]:


cross_tab1 = pd.crosstab(df['income'],df['gender'])
cross_tab1


# In[37]:


from scipy.stats import chi2_contingency

stat, p, dof, expected = chi2_contingency(cross_tab1)

alpha = 0.05
print("p value = {}".format(p))
if p <= alpha:
    print("{} <= {}".format(p,alpha))
    print("there is  relationship between income and gender. So accept the  alternative Hypothesis")
else:
    print("there is no relationship between income and gender. So reject the  Null Hypothes")


# # Step - 7 - Split the data into train and test. After which you need to perform feature transformation:
# 

# In[38]:


x, y = df.iloc[:, :-1], df.iloc[:, -1]
categorical_index = x.select_dtypes(include=['object']).columns
numerical_index = x.select_dtypes(include=['int64', 'float64']).columns
y = LabelEncoder().fit_transform(y)


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

X_train.head()


# In[40]:


y_train


# In[41]:


categorical_index


# In[42]:


X_train.dtypes

X_train_categorical = X_train.select_dtypes(include=['object'])
X_train_categorical.head()

X_train_numerical = X_train.select_dtypes(include=[ 'float64',"int64"])
X_train_numerical.head()


# In[43]:


X_train_categorical.head()


# In[44]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
X_train_categorical['workclass']= label_encoder.fit_transform(X_train_categorical['workclass'])
X_train_categorical['education']= label_encoder.fit_transform(X_train_categorical['education'])
X_train_categorical['marital-status']= label_encoder.fit_transform(X_train_categorical['marital-status'])
X_train_categorical['occupation']= label_encoder.fit_transform(X_train_categorical['occupation'])
X_train_categorical['relationship']= label_encoder.fit_transform(X_train_categorical['relationship'])
X_train_categorical['race']= label_encoder.fit_transform(X_train_categorical['race'])
X_train_categorical['gender']= label_encoder.fit_transform(X_train_categorical['gender'])
X_train_categorical['native-country']= label_encoder.fit_transform(X_train_categorical['native-country'])
X_train_categorical


# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_numerical_rescaled = pd.DataFrame(scaler.fit_transform(X_train_numerical), 
                                    columns = X_train_numerical.columns, 
                                    index = X_train_numerical.index)
X_train_numerical_rescaled.head()

X_train_transformed = pd.concat([X_train_numerical_rescaled, X_train_categorical], axis=1)

X_train_transformed.head()


# In[46]:


X_test.dtypes

X_test_categorical = X_test.select_dtypes(include=['object'])
X_test_categorical.head()

X_test_numerical = X_test.select_dtypes(include=[ 'float64',"int64"])
X_test_numerical.head()


# In[47]:


label_encoder = preprocessing.LabelEncoder()
X_test_categorical['workclass']= label_encoder.fit_transform(X_test_categorical['workclass'])
X_test_categorical['education']= label_encoder.fit_transform(X_test_categorical['education'])
X_test_categorical['marital-status']= label_encoder.fit_transform(X_test_categorical['marital-status'])
X_test_categorical['occupation']= label_encoder.fit_transform(X_test_categorical['occupation'])
X_test_categorical['relationship']= label_encoder.fit_transform(X_test_categorical['relationship'])
X_test_categorical['race']= label_encoder.fit_transform(X_test_categorical['race'])
X_test_categorical['gender']= label_encoder.fit_transform(X_test_categorical['gender'])
X_test_categorical['native-country']= label_encoder.fit_transform(X_test_categorical['native-country'])
X_test_categorical


# In[48]:


X_test_numerical_rescaled = pd.DataFrame(scaler.fit_transform(X_test_numerical), 
                                    columns = X_test_numerical.columns, 
                                    index = X_test_numerical.index)
X_test_numerical_rescaled.head()

X_test_transformed = pd.concat([X_test_numerical_rescaled, X_test_categorical], axis=1)

X_test_transformed.head()


# # Step - 8 - Build various Machine Learning models considering ‘income’ as target variable. Also make sure to perform Hyperparameter tuning to avoid Overfitting of models. 
# 

# In[49]:


from sklearn.linear_model import LogisticRegression

clf_logit = LogisticRegression()

clf_logit.fit(X_train_transformed, y_train)


# In[50]:


y_test_pred = clf_logit.predict(X_test_transformed)


# In[51]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)


# In[52]:


metrics.plot_confusion_matrix(clf_logit, X_test_transformed, y_test)


# In[53]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_transformed, y_train)
y_test_pred = nb_classifier.predict(X_test_transformed)
metrics.accuracy_score(y_test, y_test_pred) 
   


# In[54]:


metrics.plot_confusion_matrix(nb_classifier, X_test_transformed, y_test)


# In[55]:


from sklearn.svm import SVC
sv_classifier = SVC


# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[58]:


model = RandomForestClassifier(n_estimators=100)
print(model)
model.fit(X_train_transformed, y_train)
y_test_pred = model.predict(X_test_transformed)
metrics.accuracy_score(y_test, y_test_pred) 


# In[59]:


metrics.plot_confusion_matrix(model, X_test_transformed, y_test)


# In[61]:


from sklearn.svm import SVC
C=0.1
sv_classifier = SVC(C=C)


# In[63]:


sv_classifier.fit(X_train_transformed, y_train)
y_test_pred = sv_classifier.predict(X_test_transformed)

metrics.accuracy_score(y_test, y_test_pred) 


# In[64]:


metrics.plot_confusion_matrix(sv_classifier, X_test_transformed, y_test)


# In[65]:


from sklearn.neighbors import KNeighborsClassifier


# In[66]:


k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_transformed, y_train)
y_test_pred = knn_classifier.predict(X_test_transformed)
metrics.accuracy_score(y_test, y_test_pred) 


# In[67]:


metrics.plot_confusion_matrix(knn_classifier, X_test_transformed, y_test)


# # Step - 9 - Create a table to compare the performance of each of the ML Model

# In[72]:


table1=[['LogisticRegression', 0.8216875411997363], ['GaussianNB', 0.8052900461437047], ['RandomForestClassifier', 0.8498681608437706], ['Support Vector Machine', 0.773071852340145], ['KNeighborsClassifier', 0.8244891232696111]]


# In[73]:


model_accuracy_table = pd.DataFrame(table1, columns=['Model', 'Accuracy'])


# # Comparision Table

# In[74]:


model_accuracy_table


# In[76]:


model_accuracy_table.describe()


# In[ ]:





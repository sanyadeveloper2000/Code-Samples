#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_excel("covid2.xlsx")


# In[3]:


df.drop('Mycoplasma pneumoniae',axis=1,inplace=True)


# In[4]:


df.fillna(df.mean(),inplace=True)


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X=df.drop(['Patient ID', 'SARS-Cov-2 exam result' ,'Patient addmited to regular ward (1=yes, 0=no)',
       'Patient addmited to semi-intensive unit (1=yes, 0=no)',
       'Patient addmited to intensive care unit (1=yes, 0=no)','Sodium'],axis=1)


# In[7]:


y=df['Patient addmited to regular ward (1=yes, 0=no)']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


lr=LogisticRegression()


# In[14]:


lr.fit(X_train,y_train)


# In[16]:


p=lr.predict(X_test)


# In[17]:


import pickle as pkl


# In[20]:


pkl.dump(lr,open('model1.pkl','wb'))


# In[21]:


model1=pkl.load(open('model1.pkl','rb'))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


data =pd.read_csv('C:/Users/lenovo/OneDrive/Documents/Social_Network_Ads.csv')


# In[3]:


data


# In[4]:


#logiestic reagression


# In[5]:


data.head()


# In[6]:


# 0= not purchesd
# 1- purchesd


# In[7]:


data.columns


# In[8]:


# 1 st data pre-processing
# 2 nd logistic regression 
#3 rd evalute


# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[10]:


# X and y are ---input and output
data.iloc[:]


# In[11]:


data.iloc[:,[2,3]].values


# In[12]:


x=data.iloc[:,[2,3]].values


# In[13]:


y=data.iloc[:,4].values


# In[14]:


x


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)


# In[18]:


x_train


# In[19]:


x_train.shape


# In[20]:


x_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


sc= StandardScaler()


# In[23]:


x_train=  sc.fit_transform(x_train)


# In[24]:


x_test=sc.fit_transform(x_test)


# In[25]:


x_test


# In[26]:


x_train


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


log_reg= LogisticRegression(random_state=0)


# In[29]:


log_reg.fit(x_test,y_test)


# In[ ]:





# In[30]:


y_pred=log_reg.predict(x_test)


# In[31]:


print(y_pred)


# In[32]:


print(y_test)


# In[33]:


data.iloc[:,2]


# In[ ]:





# In[34]:


x_test[:,0]


# In[35]:


plt.scatter(x_test[:,0],y_test,c=y_pred)


# In[36]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[37]:


print("Accuracy -",accuracy_score(y_test,y_pred))


# In[38]:


cf=confusion_matrix(y_test,y_pred)


# In[39]:


cf


# 

# In[40]:


cl_r=classification_report(y_test,y_pred)


# In[41]:


cl_r


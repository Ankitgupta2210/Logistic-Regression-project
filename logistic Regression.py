#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[ ]:





# In[3]:


data =pd.read_csv('C:/Users/lenovo/OneDrive/Documents/Social_Network_Ads.csv')


# In[4]:


data


# In[5]:


#logiestic reagression


# In[6]:


data.head()


# In[7]:


# 0= not purchesd
# 1- purchesd


# In[8]:


data.columns


# In[9]:


# 1 st data pre-processing
# 2 nd logistic regression 
#3 rd evalute


# In[10]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


# X and y are ---input and output
data.iloc[:]


# In[14]:


data.iloc[:,[2,3]].values


# In[18]:


x=data.iloc[:,[2,3]].values


# In[19]:


y=data.iloc[:,4].values


# In[20]:


x


# In[22]:


y


# In[17]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[22]:


x_train


# In[23]:


x_train.shape


# In[24]:


x_test.shape


# In[ ]:





# In[28]:


from sklearn.preprocessing import StandardScaler

C
# In[ ]:





# In[35]:


sc= StandardScaler()


# In[37]:


x_train=  sc.fit_transform(x_train)


# In[38]:


x_test=sc.fit_transform(x_test)


# In[39]:


x_test


# In[40]:


x_train


# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


log_reg= LogisticRegression(random_state=0)


# In[43]:


log_reg.fit(x_test,y_test)


# In[ ]:





# In[44]:


y_pred=log_reg.predict(x_test)


# In[45]:


print(y_pred)


# In[46]:


print(y_test)


# In[48]:


data.iloc[:,2]


# In[52]:


plt.scatter(x_test.iloc[:,2],y_test)


# In[55]:


x_test[:,0]


# In[58]:


plt.scatter(x_test[:,0],y_test,c=y_pred)


# In[62]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[63]:


print("Accuracy -",accuracy_score(y_test,y_pred))


# In[66]:


cf=confusion_matrix(y_test,y_pred)


# In[67]:


cf


# 

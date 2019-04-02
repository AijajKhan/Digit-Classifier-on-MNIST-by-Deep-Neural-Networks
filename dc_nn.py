#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid",color_codes=True)
import pandas as pd
import keras


# In[6]:


from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# In[10]:


df=pd.read_csv("C:\\Users\\Admin\Desktop\\digits classifier\\using keras\\train.csv")


# In[22]:


x_test=pd.read_csv("C:\\Users\\Admin\Desktop\\digits classifier\\using keras\\test.csv")


# In[12]:


df.info()


# In[16]:


df.head(3)


# In[33]:


y_trainx=df.label
x_train=df.drop("label",axis=1)


# In[43]:


y_train=to_categorical(y_trainx)


# In[44]:


#print(y_train)


# In[61]:


model=Sequential()
model.add(Dense(init="uniform",input_dim=784,output_dim=60,activation="relu"))


# In[62]:


model.add(Dense(init="uniform",output_dim=60,activation="relu"))

model.add(Dense(init="uniform",output_dim=50,activation="relu"))

model.add(Dense(init="uniform",output_dim=30,activation="relu"))

model.add(Dense(init="uniform",output_dim=10,activation="softmax"))


# In[72]:


monitoring=EarlyStopping(patience=5)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
monitoring=EarlyStopping(patience=5)


# In[67]:


model.fit(x_train,y_train,epochs=20,callbacks=[monitoring],validation_split=0.2)


# In[76]:


predictions=model.predict_classes(x_test)


# In[82]:


pd.DataFrame({"ImageId": range(1,len(predictions)+1,1), "Label": predictions}).to_csv("mak.csv",index=False)


# In[ ]:





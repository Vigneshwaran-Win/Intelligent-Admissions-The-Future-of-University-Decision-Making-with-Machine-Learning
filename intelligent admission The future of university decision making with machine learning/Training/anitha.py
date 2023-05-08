#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


data = pd.read_csv(r"C:\Users\vigne\Intelligent Admissions The Future of University Decision Making with Machine Learning\Dataset\Admission_Predict.csv")


# In[17]:


data.info()


# In[18]:


data.isnull().any()


# In[23]:


data = data.rename(columns = {'Chance of Admit':'Chance of Admit'})


# In[24]:


data.describe()


# In[27]:


sns.distplot(data['GRE Score'])


# In[28]:


sns.pairplot(data=data,hue='Research',markers=["^","v"],palette='inferno')


# In[29]:


sns.scatterplot(x='University Rating',y='CGPA',data=data,color='Red',s=100)


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the data
data = pd.DataFrame({
    'GRE Score': np.random.randint(300, 340, size=100),
    'TOEFL Score': np.random.randint(90, 120, size=100),
    'University Rating': np.random.randint(1, 6, size=100),
    'SOP': np.random.uniform(1.0, 5.0, size=100),
    'LOR': np.random.uniform(1.0, 5.0, size=100),
    'CGPA': np.random.uniform(6.0, 10.0, size=100),
    'Research': np.random.randint(0, 2, size=100),
    'Chance of Admit': np.random.uniform(0.0, 1.0, size=100)
})

# Define the categories and colors
category = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']
color = ['yellowgreen', 'gold', 'lightskyblue', 'pink', 'red', 'purple', 'orange', 'gray']

# Create the subplots
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 8))

for i in range(8):
    row = i // 2
    col = i % 2
    axs[row, col].hist(data[category[i]], color=color[i], bins=10)
    axs[row, col].set_title(category[i])

plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()


# In[33]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x=sc.fit_transform(x)
x


# In[15]:


x=data.iloc[:,0:7].values
x


# In[16]:


y=data.iloc[:,7:].values
y


# In[15]:


from sklearn.model_selection import train_test_split
x = [[0],[1],[2],[3]]
y = [0, 1, 2, 3]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)


# In[18]:


y_train=(y_train>0.5)
y_train


# In[34]:


y_test=(y_test>0.5)


# In[ ]:


#train_data = pd.read_csv(r"C:\Users\vigne\Intelligent Admissions The Future of University Decision Making with Machine Learning\Dataset\Admission_Predict.csv")
#x_train = train_data
#y_train = train_data


# In[1]:


from sklearn.linear_model.logistic import LogisticRegression cls =LogisticRegression(random_state = 2)
lr=cls.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)


# In[2]:


y_pred =lr.predict(x_test)
print(y_pred)


# In[3]:


#libraries to train neural network 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


# In[4]:


model=keras.Sequential()
model.add(Dense(7,activation ='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()


# In[ ]:


#from sklearn.linear_model import LinearRegression

#model = LinearRegression()
#model.fit(x_train, y_train)
model.fit(x_train, y_train, batch_size = 20, epochs = 100)


# In[44]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy']) 


# In[5]:


model.fit(x_train, y_train, batch_size = 20, epochs = 100)


# In[6]:


from sklearn.metrics import accuracy_score 

train_prediction = model.predict(x_train)

print(train_predictions)


# In[10]:


print(classification report(y test.pred))


# In[8]:


train_acc = model.evaluate(x_train, y_train, verbose=0)[1]

print(train_acc)


# In[9]:


train_acc = model.evaluate(x_test, y_test, verbose=0)[1]

print(train_acc)


# In[11]:


pred=model.predict(x_test)
pred = (pred>0.5)
pred 


# In[12]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\nAccuracy score: %f" %(accuracy_score()*100))
print("Recall score : %f" %(recall_score(y_test,y_pred)*100))
print("ROC score : %f\n" %(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# In[13]:


#ANN Model 
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matr: 
print(classification_report(y_train,pred))


# In[14]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score 

print(classification_report(y_test,pred))


# In[15]:


model.save('model.h5')


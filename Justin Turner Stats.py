#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import sklearn.linear_model as lm

import sklearn
import sklearn.datasets
import sklearn.linear_model


# In[3]:


jt_2014 = pd.read_html('Player Data/jt-batting-2014.xls', skiprows = 0)[0]
jt_2015 = pd.read_html('Player Data/jt-batting-2015.xls', skiprows = 0)[0]
jt_2016 = pd.read_html('Player Data/jt-batting-2016.xls', skiprows = 0)[0]
jt_2017 = pd.read_html('Player Data/jt-batting-2017.xls', skiprows = 0)[0]
jt_2018 = pd.read_html('Player Data/jt-batting-2018.xls', skiprows = 0)[0]
jt_2019 = pd.read_html('Player Data/jt-batting-2019.xls', skiprows = 0)[0]


# In[4]:


# Removes columns 36-37 and the final row,served as table averages but not needed here
jt_2014.drop(index = 109, columns = ['Unnamed: 36', 'Unnamed: 37'],inplace = True)

# Removes final row that took column averages
# Also removing two columns which listed fantasy points recorded in 15'-19' seasons
jt_2015.drop(index = 126, columns = ['DFS(DK)', 'DFS(FD)'],inplace = True)
jt_2016.drop(index = 151, columns = ['DFS(DK)', 'DFS(FD)'],inplace = True)
jt_2017.drop(index = 130, columns = ['DFS(DK)', 'DFS(FD)'],inplace = True)
jt_2018.drop(index = 103, columns = ['DFS(DK)', 'DFS(FD)'],inplace = True)
jt_2019.drop(index = 135, columns = ['DFS(DK)', 'DFS(FD)'],inplace = True)


# In[5]:


# Some values were left empty in downloaded data,cross reference to retrosheet.org data for inserted values
# https://www.retrosheet.org/boxesetc/2014/Iturnj0010072014.htm
# Will not change aLI Nan value since it won't be used
jt_2014.at[28,'IBB'] = 0
jt_2014.at[28,'SF'] = 0
jt_2014.at[28,'GDP'] = 0


# In[6]:


# Some values were left empty in downloaded data,cross reference to retrosheet.org data for inserted values
# https://www.retrosheet.org/boxesetc/2018/Iturnj0010112018.htm
# Will not change aLI, WPA, RE24 Nan value since it won't be used
jt_2018.at[19,'IBB'] = 0
jt_2018.at[19,'SF'] = 0
jt_2018.at[19,'GDP'] = 0


# In[7]:


jt_5y = pd.concat([jt_2014, jt_2015, jt_2016, jt_2017, jt_2018,jt_2019], ignore_index = True)


# In[8]:


# Renames an Unnamed column to Location
def rename_unnamed_column(x):
    x.rename(columns = {'Unnamed: 5':'Location'}, inplace = True)


# In[9]:


rename_unnamed_column(jt_5y)


# In[10]:


# Maps @ to away, Nan to Home for Location column
def location_fun(x):
    x['Location'] = x['Location'].map({'@': 'Away', np.nan: 'Home'})


# In[11]:


location_fun(jt_5y)


# In[12]:


# Converts W = 1 & L = 0 for use in the model
jt_5y['Outcome'] = jt_5y['Rslt'].astype(str).str[0]
jt_5y['Outcome'] = jt_5y['Outcome'].map({'W': 1, 'L': 0})


# In[13]:


# Converts Home = 1 & Away = 0 for use in the model
jt_5y['Encoded_Location'] = jt_5y['Location'].map({ 'Home': 1 , 'Away': 0 })


# In[14]:


# Missing data, 1 means average pressure
jt_5y.at[28,'aLI'] = 1
jt_5y.at[535,'aLI'] = 1


# In[17]:


sns.boxplot(x = jt_5y.sort_values('BA')['Opp'], y = jt_5y.sort_values('BA')['BA']);
plt.rcParams['figure.figsize']=(10,10);
sns.set(font_scale= 1);
plt.xticks(rotation=90);
plt.title('Justin Turner: Batting Average By Opponent');
plt.xticks(rotation = 45);


# In[ ]:





# In[56]:


# Define our features/target
X = jt_5y[['PA','AB','R','H','2B','RBI', 'Encoded_Location','HR']]
# Target data['target'] = 0 is malignant 1 is benign
Y = (jt_5y['Outcome'] == 1).values


# In[57]:


# Split between train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)

print(f"Training Data Size: {len(x_train)}")
print(f"Test Data Size: {len(x_test)}")


# In[58]:


lr = sklearn.linear_model.LogisticRegression(fit_intercept=True)

lr.fit(x_train,y_train) # SOLUTION
train_accuracy = sum(lr.predict(x_train) == y_train) / len(y_train) # SOLUTION
test_accuracy = sum(lr.predict(x_test) == y_test) / len(y_test) # SOLUTION

print(f"Train accuracy: {train_accuracy:.6f}")
print(f"Test accuracy: {test_accuracy:.6f}")

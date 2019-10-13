#!/usr/bin/env python
# coding: utf-8

# In[142]:


import seaborn as sns
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import ds100_utils
from ds100_utils import fetch_and_cache
from pandas import ExcelFile
from pandas import ExcelWriter


# In[143]:


mlb_batters_2018 = pd.read_csv('mlb-player-stats-Batters.csv')
mlb_dh_2018 = pd.read_csv('mlb-player-stats-DH.csv')
mlb_pitchers_2018 = pd.read_csv('mlb-player-stats-P.csv')
#data from rotowire.com


# In[144]:


mlb_batters_2018.head()


# In[145]:


zf = pd.read_html('sportsref_download.xls', skiprows = 0)[0]
zf.head()


# In[154]:


JT_2014 = zf.drop(index = [109], columns = ['Unnamed: 36', 'Unnamed: 37'])


# In[158]:


JT_2014.rename(columns = {"Unnamed: 5":"Location"}, inplace = True)


# In[168]:


JT_2014['Location'] = JT_2014['Location'].map({'@':'Away', np.nan : 'Home'})


# In[169]:


JT_2014


# In[188]:


JT_2014['Date']


# In[ ]:





# In[ ]:





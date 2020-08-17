#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_columns',None)
NYC_COUNTIES = ['Bronx County', 'Kings County', 'New York County', 'Queens County', 'Richmond County']


# In[2]:


data1990f = pd.read_excel('../data/LTDB_Full_1990.xlsx')
data1990f = data1990f[data1990f.state == 'NY'][data1990f.county.isin(NYC_COUNTIES)]

data1990s = pd.read_excel('../data/LTDB_Sample_1990.xlsx')
data1990s = data1990s[data1990s.state == 'NY'][data1990s.county.isin(NYC_COUNTIES)]

data1990 = data1990f.merge(data1990s, how='inner')
del data1990f, data1990s


# In[3]:


data2000f = pd.read_excel('../data/LTDB_Full_2000.xlsx')
data2000f = data2000f[data2000f.state == 'NY'][data2000f.county.isin(NYC_COUNTIES)]

data2000s = pd.read_excel('../data/LTDB_Sample_2000.xlsx')
data2000s = data2000s[data2000s.state == 'NY'][data2000s.county.isin(NYC_COUNTIES)]

data2000 = data2000f.merge(data2000s, how='inner')
del data2000f, data2000s


# In[4]:


data2010f = pd.read_excel('../data/LTDB_Full_2010.xlsx')
data2010f = data2010f[data2010f.state == 'NY'][data2010f.county.isin(NYC_COUNTIES)]

data2010s = pd.read_excel('../data/LTDB_Sample_2010.xlsx')
data2010s = data2010s[data2010s.tractid.isin(data2010f.tractid.values)]

data2010 = data2010f.merge(data2010s, how='inner')
del data2010f, data2010s


# In[6]:


columns1990 = list(data1990.columns)
columns2000 = list(data2000.columns)
columns2010 = list(data2010.columns)


# In[7]:


columns1990 = {c:c.replace('90','').lower() for c in columns1990 if isinstance(c,str)}
columns2000 = {c:c.replace('00','').lower() for c in columns2000 if isinstance(c,str)}
columns2010 = {c:c.replace('12','').replace('10','').lower() for c in columns2010 if isinstance(c,str)}
columns2010['tractid'] = 'trtid10'


# In[8]:


data1990 = data1990.rename(columns=columns1990)
data2000 = data2000.rename(columns=columns2000)
data2010 = data2010.rename(columns=columns2010)


# In[9]:


columns = set(data1990.columns) & set(data2000.columns) & set(data2010.columns)


# In[10]:


data2000 = data2000[columns]
data1990 = data1990[columns]
data2010 = data2010[columns]


# In[18]:


tracts = set(data2000.trtid10.values) & set(data2010.trtid10.values) & set(data1990.trtid10.values)


# In[21]:


data2000 = data2000[data2000.trtid10.isin(tracts)].to_csv('data2000.csv')
data1990 = data1990[data1990.trtid10.isin(tracts)].to_csv('data1990.csv')
data2010 = data2010[data2010.trtid10.isin(tracts)].to_csv('data2010.csv')


# In[ ]:





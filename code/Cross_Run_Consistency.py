#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import sys
sys.path.insert(0,'../code/')
from shared import defaultdict,tqdm,pd,np,resultpath


# In[2]:


K = 5
F = 5
results = defaultdict(dict)
for k in range(2, K+1):
    for f in range(2, F+1):
        with open(f'../results/kmeans/rawresults/result_{k}{f}.pickle','rb') as p:
            results[k][f] = pickle.load(p)[0]
            
tracts = results[2][2][0].index.values.tolist()


# In[ ]:


inconsistency = defaultdict(dict)
for k in tqdm(range(2, K+1)):
    for f in tqdm(range(2, F+1)):
        r = results[k][f]
        cooc = defaultdict(lambda: defaultdict(list))
        for tracti in tqdm(tracts):
            for tractj in tracts:
                for i in range(10):
                    mi = r[i][r[i].index == tracti].model.values[0]
                    mj = r[i][r[i].index == tractj].model.values[0]
                    cooc[tracti][tractj].append(1 if mi == mj else 0)
        scores = []
        for tracti,subd in tqdm(cooc.items()):
            for tractj, v in subd.items():
                match = len([t for t in v if t == 1])
                nomatch = len([t for t in v if t == 0])
                scores.append(
                    (match/nomatch) if match <= nomatch else (nomatch/match)
                )
        inconsistency[k][f] = np.mean(scores)


# In[ ]:

with open(f'{resultpath}inconsistency.pickle','wb') as f:
    pickle.dump(inconsistency,f)


# In[ ]:





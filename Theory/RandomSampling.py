#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:16:04 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
# %% sampling Range

b1=0
b2=np.pi

# %% Local Minima
LocalMinimas=np.load('./LocalMinima/LocalMinimaMichalewicz.npy', allow_pickle=True)




#%% define random search

def random_search_(number_samples, T, bd1, bd2, LocalMinima, dim, eps):
    
    NumberOfFoundLocalMinima=np.zeros(T)
    score=np.zeros(len(LocalMinima))
    for t in range(T):
        
        # generate number_samples samples
        
        Samples=np.random.uniform(bd1, bd2, size=(number_samples, dim))
        
        # check how many local minima found 
        
        for k in range(len(LocalMinima)):
            # check for lall ocal minima that were not found so far if some particle i foudn it at time t
            if score[k]==0:
                # check for each particle
                for  i in range(number_samples):
                    if np.linalg.norm(Samples[i,:]-LocalMinima[k])<eps:
                        score[k]=1
                        break
        
        NumberOfFoundLocalMinima[t]=np.count_nonzero(score)/len(LocalMinima)
    
    return NumberOfFoundLocalMinima
            
            
        
        
#%% 
eps=0.52
number_samples=20
T=200
simulations=2
dim=5
NumberOfFoundLocalMinima=np.zeros(T)
#%% Rastrigin
for i in range(simulations):
    NumberOfFoundLocalMinima=random_search_(number_samples, T,b1,b2, LocalMinimas, dim, eps)+NumberOfFoundLocalMinima
    
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinima/simulations)

#%%
#np.save('./Results/Random',NumberOfFoundLocalMinima/200)

#%%
'''
Eps=np.arange(0,1+0.1,0.1)

NF=np.zeros(10)

for i in range(len(Eps)-1): # loop of different radi
    NF_i=random_search_(number_samples, simulations,b1,b2, LocalMinimas, dim, Eps[i+1])
    NF[i]=NF_i[-1]
    print(NF[i])
#%%

fig=plt.figure()
plt.plot(Eps[1:11],NF)
plt.plot(Eps[1:11],AverageExplorationDivergent/120,color='blue')
'''
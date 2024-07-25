#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:16:04 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import *
# %% sampling Range

b1=0
b2=np.pi

# %% Local Minima
LocalMinimas=np.load('./LocalMinima/LocalMinimaMichalewicz.npy', allow_pickle=True)

'''


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
simulations=200 # ovver how many we average
dim=5
NumberOfFoundLocalMinima=np.zeros(T)
#%% Rastrigin
for i in range(simulations):
    NumberOfFoundLocalMinima=random_search_(number_samples, T,b1,b2, LocalMinimas, dim, eps)+NumberOfFoundLocalMinima
#%%
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinima/simulations*120)

#%%
#np.save('./Results/Random',NumberOfFoundLocalMinima/200)

#%%

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

#%%%

#%% define random search
from scipy.optimize import minimize
def random_search_2(number_samples, T, bd1, bd2, LocalMinima, dim, eps, sim):
    
    NumberOfFoundLocalMinima=np.zeros(sim)
    score=np.zeros(len(LocalMinima))
    print(sim)
    for s in range(sim):
        print(s)
        

        # generate number_samples samples
        
        Samples=np.random.uniform(bd1, bd2, size=(number_samples*T, dim))
        
        # check how many local minima found 
        
        for k in range(len(LocalMinima)):
            # check for lall ocal minima that were not found so far if some particle i foudn it at time t
            if score[k]==0:
                # check for each particle
                for  i in range(number_samples*T):
                    #res = minimize(fct_michalewicz,Samples[i,:], method='CG',options={'disp': False, 'maxiter':100})
                    #sol=res.x
                    
                    if np.linalg.norm(Samples[i,:]-LocalMinima[k])<eps:
                    #if np.linalg.norm(sol-LocalMinima[k])<eps:
                        print('yes',i,k)
                        score[k]=1
                        break
    
        NumberOfFoundLocalMinima[s]=np.count_nonzero(score)/len(LocalMinima)
        print(np.max(NumberOfFoundLocalMinima))
            
            
    
    return NumberOfFoundLocalMinima
            
            
        
        
#%% 
Eps=np.arange(0.05,0.4,0.025)

number_samples=20
T=200
simulations=1 # ovver how many we average
dim=5

NumberOfFoundLocalMinima=np.zeros(14)
#%% Rastrigin
k=0
for eps in Eps:
    print(eps)
    for s in range(200):
        print(s)
        NumberOfFoundLocalMinima[k]=random_search_2(number_samples, T,b1,b2, LocalMinimas, dim, eps, simulations)+NumberOfFoundLocalMinima[k]
    
    k=k+1
#%%


np.save('./Results/Random',NumberOfFoundLocalMinima*120/200)
ddd
#%%
simulations=200
NumberOfFoundLocalMinima=random_search_2(number_samples, 20,b1,b2, LocalMinimas, dim, 0.52, 200)
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinima*120)
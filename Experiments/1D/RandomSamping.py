#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:51:02 2024

@author: bzfkostr
"""
''' Random Sampling'''

import numpy as np
import matplotlib.pyplot as plt

# %% sampling Range

X1=[-100,-5.12, -30, -20, -100,0, -400, -600] # lower range
X2=[100,5.12, 30,20, 100, 10, 400, 600]       # upper range


# %% Local Minima


LocalMinimaRastrigin=np.load('./LocalMinima/LocalMinimaRastrigin.npy') # fct number 1
LocalMinimaHM=np.load('./LocalMinima/LocalMinimaHM.npy') # fct number 5
LocalMinimaSchwefel=np.load('./LocalMinima/LocalMinimaSchwefel.npy') # fct number 6
LocalMinimaGriewank=np.load('./LocalMinima/LocalMinimaGriewank.npy') # fct number 7

#%% define random search

def random_search_(number_samples, simulations, bd1, bd2, LocalMinima, dim, eps):
    
    NumberOfFoundLocalMinima=np.zeros(simulations)
    score=np.zeros(len(LocalMinima))
    for s in range(simulations):
        
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
        
        NumberOfFoundLocalMinima[s]=np.count_nonzero(score)/len(LocalMinima)
    
    return NumberOfFoundLocalMinima
            
            
        
        
#%% 
eps=0.1
number_samples=20
simulations=20
dim=1
#%% Rastrigin

NumberOfFoundLocalMinimaRastrigin=random_search_(number_samples, simulations,-5.12, 5.12, LocalMinimaRastrigin, dim, eps)
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaRastrigin)

#%% Hole

NumberOfFoundLocalMinimaHole=random_search_(1000, simulations,0, 11, LocalMinimaHM, dim, eps)
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaHole)

#%% Schwefel

NumberOfFoundLocalMinimaSchwefel=random_search_(1000, simulations,-400, 400, LocalMinimaSchwefel, dim, eps)
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaSchwefel)

#%%Griewank

NumberOfFoundLocalMinimaGriewank=random_search_(1000, simulations,-600,600, LocalMinimaGriewank, dim, eps)
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaGriewank)
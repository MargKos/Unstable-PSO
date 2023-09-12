#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:24:51 2023

@author: bzfkostr
"""

import numpy as np
from Rastrigin_fct import *
from Variables import *

'''
Avverage exploration over all simulations, saves all local minima in one file and calculates their function value
'''

# averages number of found local minima from multiprocessing simulatios and calculates the function value of found local minima

AverageNumberOfFoundLocalMinimaOverdamped=np.zeros(T_PSO) # average number of found local minima for overdamped PSO
AverageNumberOfFoundLocalMinimaDamped=np.zeros(T_PSO) # average number of found local minima for damped PSO
AverageNumberOfFoundLocalMinimaDivergent=np.zeros(T_PSO) # average number of found local minima for divergent PSO
for s  in range(sim):
    
    # load number of found local minima
    NumberOfFoundLocalMinimaOverdamped=np.load(path+'NumberOfFoundLocalMinima'+str(nameOverdamped)+str(s)+'.npy')
    NumberOfFoundLocalMinimaDivergent=np.load(path+'NumberOfFoundLocalMinima'+str(nameDivergent)+str(s)+'.npy')
    NumberOfFoundLocalMinimaDamped=np.load(path+'NumberOfFoundLocalMinima'+str(nameDamped)+str(s)+'.npy')

    AverageNumberOfFoundLocalMinimaOverdamped+=NumberOfFoundLocalMinimaOverdamped
    AverageNumberOfFoundLocalMinimaDamped+=NumberOfFoundLocalMinimaDamped
    AverageNumberOfFoundLocalMinimaDivergent+=NumberOfFoundLocalMinimaDivergent

# averages over all simulations
AverageNumberOfFoundLocalMinimaOverdamped=AverageNumberOfFoundLocalMinimaOverdamped/sim
AverageNumberOfFoundLocalMinimaDamped=AverageNumberOfFoundLocalMinimaDamped/sim
AverageNumberOfFoundLocalMinimaDivergent=AverageNumberOfFoundLocalMinimaDivergent/sim

#save files

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameOverdamped)+'', AverageNumberOfFoundLocalMinimaOverdamped)

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameDamped)+'', AverageNumberOfFoundLocalMinimaDamped)

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameDivergent)+'', AverageNumberOfFoundLocalMinimaDivergent)



#%%
AllLocalMinimaOverdamped=[] # all local minima found by overdamped PSO
AllLocalMinimaDivergent=[] # all local minima found by divergent PSO
AllLocalMinimaDamped=[] # all local minima found by damped PSO
for s in range(sim):
    LocalMinimaOverdamped=np.load(path+'LocalMinima'+str(nameOverdamped)+str(s)+'.npy')
    LocalMinimaDivergent=np.load(path+'LocalMinima'+str(nameDivergent)+str(s)+'.npy')
    LocalMinimaDamped=np.load(path+'LocalMinima'+str(nameDamped)+str(s)+'.npy')

    for i in range(len(LocalMinimaOverdamped)):
        AllLocalMinimaOverdamped.append(LocalMinimaOverdamped[i])
    
    for i in range(len(LocalMinimaDamped)):
        AllLocalMinimaDamped.append(LocalMinimaDamped[i])
    
    for i in range(len(LocalMinimaDivergent)):
        AllLocalMinimaDivergent.append(LocalMinimaDivergent[i])
    
np.save('./Results/AllLocalMinima'+str(nameOverdamped)+'.npy', AllLocalMinimaOverdamped)
np.save('./Results/AllLocalMinima'+str(nameDamped)+'.npy',AllLocalMinimaDamped)
np.save('./Results/AllLocalMinima'+str(nameDivergent)+'.npy', AllLocalMinimaDivergent)


AllLocalMinimaOverdamped=np.load('./Results/AllLocalMinima'+str(nameOverdamped)+'.npy')
AllLocalMinimaDamped=np.load('./Results/AllLocalMinima'+str(nameDamped)+'.npy')
AllLocalMinimaDivergent=np.load('./Results/AllLocalMinima'+str(nameDivergent)+'.npy')

#%% Calculate the function value of the local minima
AllLocalMinimaOverdampedValue=np.zeros(len(AllLocalMinimaOverdamped)) # function value of local minima found by overdamped PSO
AllLocalMinimaDampedValue=np.zeros(len(AllLocalMinimaDamped)) # function value of local minima found by damped PSO
AllLocalMinimaDivergentValue=np.zeros(len(AllLocalMinimaDivergent)) # function value of local minima found by dviergent PSO
for i in range(len(AllLocalMinimaOverdamped)):  
    AllLocalMinimaOverdampedValue[i]=fct_Rastrigin(AllLocalMinimaOverdamped[i])
for i in range(len(AllLocalMinimaDamped)):
    AllLocalMinimaDampedValue[i]=fct_Rastrigin(AllLocalMinimaDamped[i])
for i in range(len(AllLocalMinimaDivergent)):
    AllLocalMinimaDivergentValue[i]=fct_Rastrigin(AllLocalMinimaDivergent[i])
    
#%% save Values
    
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameDivergent)+'.npy', AllLocalMinimaDivergentValue)
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameDamped)+'.npy', AllLocalMinimaDampedValue)
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameOverdamped)+'.npy', AllLocalMinimaOverdampedValue)

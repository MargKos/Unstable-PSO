#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:24:51 2023

@author: bzfkostr
"""

import numpy as np
from Rastrigin_fct import *
from Variables import *

# averages number of found local minima and calculates their function value

AverageNumberOfFoundLocalMinimaC=np.zeros(T_PSO)
AverageNumberOfFoundLocalMinimaH=np.zeros(T_PSO)
AverageNumberOfFoundLocalMinimaO=np.zeros(T_PSO)
for s  in range(sim):
    NumberOfFoundLocalMinimaC=np.load(path+'NumberOfFoundLocalMinima'+str(nameClassic)+str(s)+'.npy')
    NumberOfFoundLocalMinimaO=np.load(path+'NumberOfFoundLocalMinima'+str(nameOrbit)+str(s)+'.npy')
    NumberOfFoundLocalMinimaH=np.load(path+'NumberOfFoundLocalMinima'+str(nameHarmonic)+str(s)+'.npy')

    AverageNumberOfFoundLocalMinimaC+=NumberOfFoundLocalMinimaC
    AverageNumberOfFoundLocalMinimaH+=NumberOfFoundLocalMinimaH
    AverageNumberOfFoundLocalMinimaO+=NumberOfFoundLocalMinimaO


AverageNumberOfFoundLocalMinimaC=AverageNumberOfFoundLocalMinimaC/sim
AverageNumberOfFoundLocalMinimaH=AverageNumberOfFoundLocalMinimaH/sim
AverageNumberOfFoundLocalMinimaO=AverageNumberOfFoundLocalMinimaO/sim

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameClassic)+'', AverageNumberOfFoundLocalMinimaC)

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameHarmonic)+'', AverageNumberOfFoundLocalMinimaH)

np.save('./Results/AverageNumberOfFoundLocalMinima'+str(nameOrbit)+'', AverageNumberOfFoundLocalMinimaO)



#%% Save local minima
AllLocalMinimaC=[]
AllLocalMinimaO=[]
AllLocalMinimaH=[]
for s in range(sim):
    LocalMinimaC=np.load(path+'LocalMinima'+str(nameClassic)+str(s)+'.npy')
    LocalMinimaO=np.load(path+'LocalMinima'+str(nameOrbit)+str(s)+'.npy')
    LocalMinimaH=np.load(path+'LocalMinima'+str(nameHarmonic)+str(s)+'.npy')

    for i in range(len(LocalMinimaC)):
        AllLocalMinimaC.append(LocalMinimaC[i])
    
    for i in range(len(LocalMinimaH)):
        AllLocalMinimaH.append(LocalMinimaH[i])
    
    for i in range(len(LocalMinimaO)):
        AllLocalMinimaO.append(LocalMinimaO[i])
    
np.save('./Results/AllLocalMinima'+str(nameClassic)+'.npy', AllLocalMinimaC)
np.save('./Results/AllLocalMinima'+str(nameHarmonic)+'.npy',AllLocalMinimaH)
np.save('./Results/AllLocalMinima'+str(nameOrbit)+'.npy', AllLocalMinimaO)


AllLocalMinimaC=np.load('./Results/AllLocalMinima'+str(nameClassic)+'.npy')
AllLocalMinimaD=np.load('./Results/AllLocalMinima'+str(nameHarmonic)+'.npy')
AllLocalMinimaO=np.load('./Results/AllLocalMinima'+str(nameOrbit)+'.npy')

#%% Calculate the function value of the local minima
AllLocalMinimaCValue=np.zeros(len(AllLocalMinimaC))
AllLocalMinimaHValue=np.zeros(len(AllLocalMinimaH))
AllLocalMinimaOValue=np.zeros(len(AllLocalMinimaO))
for i in range(len(AllLocalMinimaC)):  
    AllLocalMinimaCValue[i]=fct_Rastrigin(AllLocalMinimaC[i])
for i in range(len(AllLocalMinimaD)):
    AllLocalMinimaHValue[i]=fct_Rastrigin(AllLocalMinimaD[i])
for i in range(len(AllLocalMinimaO)):
    AllLocalMinimaOValue[i]=fct_Rastrigin(AllLocalMinimaO[i])
    
#%% save Values
    
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameOrbit)+'.npy', AllLocalMinimaOValue)
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameHarmonic)+'.npy', AllLocalMinimaHValue)
np.save('./Results/ValuesLocalMinimaRastrigin'+str(nameClassic)+'.npy', AllLocalMinimaCValue)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:37:30 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from Rastrigin_fct import *
from sklearn.manifold import TSNE
from Variables import *
from scipy.optimize import minimize
#%% Mean Divergent 
delta=5000 # calculates every delta time steps the average function value
T_PSO_long=500000 # until which time step the average function value is calculated

sim=100
LossDivergent=np.zeros(int(T_PSO_long/delta)) # gives the average function value at each time step

#%%
for t in range(int(T_PSO_long/delta)):
    List=[]
   
    for s in range(sim):
        
        DivergentG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(DivergentG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossDivergent[t]=np.mean(List) # calculate average function value at time t

#np.save('./Results/LongMeanRastrigin'+str(nameDivergent)+'', LossDivergent)

#%% Mean  Damped
        
LossDamped=np.zeros(int(T_PSO_long/delta)) # gives the average function value at each time step
for t in range(int(T_PSO_long/delta)):
    List=[]
    for s in range(sim):
        DampedG=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(DampedG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossDamped[t]=np.mean(List) # calculate average function value at time t
   
#np.save('./Results/LongMeanRastrigin'+str(nameDamped)+'', LossDamped)

#%% Mean Overdamped
        
LossOverdamped=np.zeros(int(T_PSO_long/delta)) # gives the average function value at each time step of Overdamped parameters
for t in range(int(T_PSO_long/delta)):
    List=[]
    for s in range(sim):
        OverdampedG=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(OverdampedG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossOverdamped[t]=np.mean(List) # calculate average function value at time t
    
#np.save('./Results/LongMeanRastrigin'+str(nameOverdamped)+'', LossOverdamped)


#%%

LossDivergent=np.load('./Results/LongMeanRastrigin'+str(nameDivergent)+'.npy')
LossDamped=np.load('./Results/LongMeanRastrigin'+str(nameDamped)+'.npy')
LossOverdamped=np.load('./Results/LongMeanRastrigin'+str(nameOverdamped)+'.npy')

X=np.arange(0, T_PSO_long, delta )

fig=plt.figure()
plt.plot(X[1:100], LossDivergent[1:100],  color='blue')
plt.plot(X[1:100], LossOverdamped[1:100],  color='orange')
plt.plot(X[1:100], LossDamped[1:100],  color='green')

#%%

DivergentS=np.zeros(sim)
for s in range(sim):
        
    DivergentG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy') # load global best positions
    DivergentS[s]=fct_Rastrigin(DivergentG[:,-1]) # calculate function value of the global best position at time t
       

DampedS=np.zeros(sim)
for s in range(sim):
        
    DampedG=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy') # load global best positions
    DampedS[s]=fct_Rastrigin(DampedG[:,-1]) # calculate function value of the global best position at time t
       
OverdampedS=np.zeros(sim)
for s in range(sim):
        
    OverdampedG=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy') # load global best positions
    OverdampedS[s]=fct_Rastrigin(OverdampedG[:,-1]) # calculate function value of the global best position at time t
       

#%%

fig = plt.figure()
plt.hist(DivergentS, alpha=0.5, label='Divergent', histtype='step')
plt.hist(DampedS, alpha=0.5, label='Damped', histtype='step')
plt.hist(OverdampedS, alpha=0.5, label='Overdamped', histtype='step')

plt.xlabel('Function Value')
plt.ylabel('Frequency')
plt.title('Histogram of Function Values')
plt.legend()
plt.show()

#%% find stagnation point of a given simulation

indices = np.where(DampedS > 350)[0]

def fct_stagnation(t, G):
    
    t1=t-50000
    t2=t
    t3=t+20000
    
    v1=fct_Rastrigin(G[:,t1])
    v2=fct_Rastrigin(G[:,t2])
    v3=fct_Rastrigin(G[:,t3])
    
    return v1,v2,v3, v3-v2

for s in indices:
    DampedG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy') # load global best positions
    
    v1,v2,v3, var=fct_stagnation(150, DampedG)

    print(v1,v2,v3, var)
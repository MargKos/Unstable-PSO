#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:42:50 2023

@author: bzfkostr
"""


import numpy as np
import matplotlib.pyplot as plt
from Rastrigin_fct import *
from sklearn.manifold import TSNE
from Variables import *
from scipy.optimize import minimize


#%% Concentration

def fct_concentration(X, Gl,  eps): #X is the matrix of all positions, Gl is the matrix of all global best positions, eps is the radius of the ball
    Concentration=np.zeros(T_PSO_short) # Concentration is the vector of the concentration at each time step
    for t in range(T_PSO_short):
        nb=0
        for i in range(n):
            if np.linalg.norm(X[:,i,t]-Gl[:,t])<eps: #if the distance between the position of the particle and the global best is smaller than eps
                nb=nb+1
    
        Concentration[t]=nb/n
    
    return Concentration

#%%

ConcentrationDamped=np.zeros(T_PSO_short) # Concentration is the vector of the concentration at each time step of the damped PSO
ConcentrationDivergent=np.zeros(T_PSO_short) # Concentration is the vector of the concentration at each time step of the divergent PSO
ConcentrationOverdamped=np.zeros(T_PSO_short) # Concentration is the vector of the concentration at each time step of the overdamped PSO
for s in range(sim):
    eps=1.3552186944893772
   

    XgDivergent=np.load(path+'G_s' + str(nameDivergent) + str(s) + '.npy') # load the global best positions of the divergent PSO
    XgOverdamped=np.load(path+'G_s' + str(nameOverdamped) + str(s) + '.npy' ) # load the global best positions of the overdamped PSO
    XgDamped=np.load(path+'G_s' + str(nameDamped) + str(s) + '.npy' ) # load the global best positions of the damped PSO
    
    XDivergent=np.load(path+'X_s' + str(nameDivergent) + str(s) + '.npy') # load the positions of the divergent PSO
    XOverdamped=np.load(path+'X_s' + str(nameOverdamped) + str(s) + '.npy') # load the positions of the overdamped PSO
    XDamped=np.load(path+'X_s' + str(nameDamped) + str(s) + '.npy') # load the positions of the damped PSO
    
    ConcentrationDivergent=ConcentrationDivergent+fct_concentration(XDivergent, XgDivergent,eps)
    ConcentrationOverdamped=ConcentrationOverdamped+fct_concentration(XOverdamped,XgOverdamped,  eps)
    ConcentrationDamped=ConcentrationDamped+fct_concentration(XDamped, XgDamped,eps)
    
np.save('./Results/AverageConcentrationRastrigin'+str(nameDivergent)+'.npy', ConcentrationDivergent/sim) # save results
np.save('./Results/AverageConcentrationRastrigin'+str(nameOverdamped)+'.npy', ConcentrationOverdamped/sim)
np.save('./Results/AverageConcentrationRastrigin'+str(nameDamped)+'.npy', ConcentrationDamped/sim)
    
print('done concentration')

#%% Mean Divergent 

delta=1
T_PSO_short=1000
LossDivergent=np.zeros(int(T_PSO_short/delta))

for t in range(int(T_PSO_short/delta)):
    
    List=[]
    for s in range(sim):
        DivergentG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy') # load the global best positions of the divergent PSO
        functionvalue=fct_Rastrigin(DivergentG[:,int(t*delta)]) # compute the function value of the global best position at time t
        List.append(functionvalue)  # add the function value to the list
        
    LossDivergent[t]=np.mean(List) # compute the mean of the function values of the global best positions at time t

np.save('./Results/ShortMeanRastrigin'+str(nameDivergent)+'', LossDivergent)
print(len(LossO))

#%% Mean Damped
        
LossDamped=np.zeros(int(T_PSO_short/delta))
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        DampedG=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(DampedG[:,int(t*delta)])
        List.append(functionvalue)
        
    LossDamped[t]=np.mean(List)
   

np.save('./Results/ShortMeanRastrigin'+str(nameDamped)+'', LossDamped)

#%% Mean Overdamped
        
LossOverdamped=np.zeros(int(T_PSO_short/delta))
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        OverdampedG=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(OverdampedG[:,int(t*delta)])
        List.append(functionvalue)
        
    LossOverdamped[t]=np.mean(List)
    
np.save('./Results/ShortMeanRastrigin'+str(nameOverdamped)+'', LossOverdamped)



#
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

def fct_concentration(X, Gl,  eps): # X is the particle positions, Gl is the global best position, eps is the radius of the ball around Gl
    Concentration=np.zeros(T_PSO_short) # gives exploitaiton at each time step
    for t in range(T_PSO_short):
        nb=0 # number of particles in the ball around Gl
        for i in range(n):
            if np.linalg.norm(X[:,i,t]-Gl[:,t])<eps: # if the particle is in the ball around Gl
                nb=nb+1
    
        Concentration[t]=nb/n # exploitaiton at time t
    
    return Concentration

#%%

ConcentrationDamped=np.zeros(T_PSO_short) # gives the average exploitaiton at each time step
ConcentrationDivergent=np.zeros(T_PSO_short) # gives the average exploitaiton at each time step
ConcentrationOverdamped=np.zeros(T_PSO_short) # gives the average exploitaiton at each time step
for s in range(sim):
    eps=1.3552186944893772 # set radius, this radius gives a volume of 0.1
    
    # load global best positions

    XgDivergent=np.load(path+'G_s' + str(nameDivergent) + str(s) + '.npy')
    XgOverdamped=np.load(path+'G_s' + str(nameOverdamped) + str(s) + '.npy' )
    XgDamped=np.load(path+'G_s' + str(nameDamped) + str(s) + '.npy' )
    
    # load particle positions

    XDivergent=np.load(path+'X_s' + str(nameDivergent) + str(s) + '.npy')
    XOverdamped=np.load(path+'X_s' + str(nameOverdamped) + str(s) + '.npy')
    XDamped=np.load(path+'X_s' + str(nameDamped) + str(s) + '.npy')
    
    # calculate exploitaiton of simulation s and add it to the average exploitaiton
    ConcentrationDivergent=ConcentrationDivergent+fct_concentration(XDivergent, XgDivergent,eps)
    ConcentrationOverdamped=ConcentrationOverdamped+fct_concentration(XOverdamped,XgOverdamped,  eps)
    ConcentrationDamped=ConcentrationDamped+fct_concentration(XDamped, XgDamped,eps)
    
np.save('./Results/AverageConcentrationRastrigin'+str(nameDivergent)+'.npy', ConcentrationDivergent/sim)
np.save('./Results/AverageConcentrationRastrigin'+str(nameOverdamped)+'.npy', ConcentrationOverdamped/sim)
np.save('./Results/AverageConcentrationRastrigin'+str(nameDamped)+'.npy', ConcentrationDamped/sim)
    
print('done concentration')

#%% Mean Divergent 
delta=1 # calculates every delta time steps the average function value
T_PSO_short=1000 # until which time step the average function value is calculated
LossDivergent=np.zeros(int(T_PSO_short/delta)) # gives the average function value at each time step

for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        DivergentG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(DivergentG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossDivergent[t]=np.mean(List) # calculate average function value at time t

np.save('./Results/ShortMeanRastrigin'+str(nameDivergent)+'', LossDivergent)

#%% Mean  Damped
        
LossDamped=np.zeros(int(T_PSO_short/delta)) # gives the average function value at each time step
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        DampedG=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(DampedG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossDamped[t]=np.mean(List) # calculate average function value at time t
   
np.save('./Results/ShortMeanRastrigin'+str(nameDamped)+'', LossDamped)

#%% Mean Overdamped
        
LossOverdamped=np.zeros(int(T_PSO_short/delta)) # gives the average function value at each time step of Overdamped parameters
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        OverdampedG=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy') # load global best positions
        functionvalue=fct_Rastrigin(OverdampedG[:,int(t*delta)]) # calculate function value of the global best position at time t
        List.append(functionvalue) # add function value to list
        
    LossOverdamped[t]=np.mean(List) # calculate average function value at time t
    
np.save('./Results/ShortMeanRastrigin'+str(nameOverdamped)+'', LossOverdamped)


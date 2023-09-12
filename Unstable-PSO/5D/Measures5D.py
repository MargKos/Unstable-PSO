#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:27:24 2023

@author: bzfkostr
"""

import numpy as np
from Functions import *
from Variables import *

# calculate the mean function value of the swarm at each time step

#%% Mean Divergent

LossDivergent=np.zeros(T_PSO) # LossDivergent gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number]
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        DivergentG=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(DivergentG[:,t])
        # append function value to list
        List.append(functionvalue)

    # calculate mean function value of each time step   
    LossDivergent[t]=np.mean(List)

# save the results
np.save('./Results/Mean'+str(nameDivergent)+'', LossDivergent)


#%% Mean Damped
        
LossDamped=np.zeros(T_PSO) # LossDamped gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number] # select the right function
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        DampedG=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(DampedG[:,t])
        # append function value to list
        List.append(functionvalue)
    # calculate mean function value of each time step    
    LossDamped[t]=np.mean(List)

# save the results
np.save('./Results/Mean'+str(nameDamped)+'', LossDamped)

#%% Mean Overdamped
        
LossOverdamped=np.zeros(T_PSO) # LossOverdamped gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number]
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        OverdampedG=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(OverdampedG[:,t])
        # append function value to list
        List.append(functionvalue)
    # calculate mean function value of each time step  
    LossOverdamped[t]=np.mean(List)
# save the results
np.save('./Results/Mean'+str(nameOverdamped)+'', LossOverdamped)


#%% Concentration of particles around globale best

def fct_concentration(X, Gl,  eps): # X=particle positions, Gl=global best position, eps=radius of ball
    Concentration=np.zeros(T_PSO) # Concentration gives the concentration of particles around the global best at time t
    for t in range(T_PSO):
        closeparticles=0 # counts the number of particles in the ball around the global best
        for i in range(n):
            if np.linalg.norm(X[:,i,t]-Gl[:,t])<eps: # check if particle i is in the ball around the global best
                closeparticles=closeparticles+1

        # calculate the concentration of particles around the global best at time t
        Concentration[t]=closeparticles/n
    
    return Concentration

#%% Average Concentration over all simulations
    
ConcentrationDamped=np.zeros(T_PSO)  # ConcentrationDamped gives the average concentration of particles around the global best at time t
ConcentrationOverdamped=np.zeros(T_PSO)   # ConcentrationOverdamped gives the average concentration of particles around the global best at time t
ConcentrationDivergent=np.zeros(T_PSO)     # ConcentrationDivergent gives the average concentration of particles around the global best at time t

for s in range(sim):    
    # load particle positions of each simulation
    X_Damped=np.load(path+'X_s'+str(nameDamped)+str(s)+'.npy')
    X_Overdamped=np.load(path+'X_s'+str(nameOverdamped)+str(s)+'.npy')
    X_Divergent=np.load(path+'X_s'+str(nameDivergent)+str(s)+'.npy')

    # load global best positions of each simulation
    G_Damped=np.load(path+'G_s'+str(nameDamped)+str(s)+'.npy')
    G_Overdamped=np.load(path+'G_s'+str(nameOverdamped)+str(s)+'.npy')
    G_Divergent=np.load(path+'G_s'+str(nameDivergent)+str(s)+'.npy')

    # calculate the concentration of particles around the global best and average
    ConcentrationDamped=ConcentrationDamped+fct_concentration(X_Damped, G_Damped, 0.52)
    ConcentrationOverdamped=ConcentrationOverdamped+fct_concentration(X_Overdamped, G_Overdamped, 0.52)
    ConcentrationDivergent=ConcentrationDivergent+fct_concentration(X_Divergent, G_Divergent, 0.52)
       
#%%
# save the results
np.save('./Results/AverageConcentration'+str(nameDivergent)+'.npy', ConcentrationDivergent/sim)
np.save('./Results/AverageConcentration'+str(nameOverdamped)+'.npy', ConcentrationOverdamped/sim)
np.save('./Results/AverageConcentration'+str(nameDamped)+'.npy', ConcentrationDamped/sim)


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

#%% Mean Orbit 

LossO=np.zeros(T_PSO) # LossO gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number]
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        OrbitG=np.load(path+'G_s'+str(nameOrbit)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(OrbitG[:,t])
        # append function value to list
        List.append(functionvalue)

    # calculate mean function value of each time step   
    LossO[t]=np.mean(List)

# save the results
np.save('./Results/Mean'+str(nameOrbit)+'', LossO)


#%% Mean Harmonic
        
LossH=np.zeros(T_PSO) # LossH gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number] # select the right function
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        HarmonicG=np.load(path+'G_s'+str(nameHarmonic)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(HarmonicG[:,t])
        # append function value to list
        List.append(functionvalue)
    # calculate mean function value of each time step    
    LossH[t]=np.mean(List)

# save the results
np.save('./Results/Mean'+str(nameHarmonic)+'', LossH)

#%% Mean Classic
        
LossC=np.zeros(T_PSO) # LossC gives the mean function value of the swarm at time t
for t in range(T_PSO):
    
    fct=FunctionList[function_number]
    List=[]
    for s in range(sim):
        # load global best position of each swarm
        ClassicG=np.load(path+'G_s'+str(nameClassic)+str(s)+'.npy', allow_pickle=True)
        # calculate function value of each global best position
        functionvalue=fct(ClassicG[:,t])
        # append function value to list
        List.append(functionvalue)
    # calculate mean function value of each time step  
    LossC[t]=np.mean(List)
# save the results
np.save('./Results/Mean'+str(nameClassic)+'', LossC)


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
    
ConcentrationHarmonic=np.zeros(T_PSO)  # ConcentrationHarmonic gives the average concentration of particles around the global best at time t
ConcentrationClassic=np.zeros(T_PSO)   # ConcentrationClassic gives the average concentration of particles around the global best at time t
ConcentrationOrbit=np.zeros(T_PSO)     # ConcentrationOrbit gives the average concentration of particles around the global best at time t

for s in range(sim):    
    # load particle positions of each simulation
    X_Harmonic=np.load(path+'X_s'+str(nameHarmonic)+str(s)+'.npy')
    X_Classic=np.load(path+'X_s'+str(nameClassic)+str(s)+'.npy')
    X_Orbit=np.load(path+'X_s'+str(nameOrbit)+str(s)+'.npy')

    # load global best positions of each simulation
    G_Harmonic=np.load(path+'G_s'+str(nameHarmonic)+str(s)+'.npy')
    G_Classic=np.load(path+'G_s'+str(nameClassic)+str(s)+'.npy')
    G_Orbit=np.load(path+'G_s'+str(nameOrbit)+str(s)+'.npy')

    # calculate the concentration of particles around the global best and average
    ConcentrationHarmonic=ConcentrationHarmonic+fct_concentration(X_Harmonic, G_Harmonic, 0.52)
    ConcentrationClassic=ConcentrationClassic+fct_concentration(X_Classic, G_Classic, 0.52)
    ConcentrationOrbit=ConcentrationOrbit+fct_concentration(X_Orbit, G_Orbit, 0.52)
       
#%%
# save the results
np.save('./Results/AverageConcentration'+str(nameOrbit)+'.npy', ConcentrationOrbit/sim)
np.save('./Results/AverageConcentration'+str(nameClassic)+'.npy', ConcentrationClassic/sim)
np.save('./Results/AverageConcentration'+str(nameHarmonic)+'.npy', ConcentrationHarmonic/sim)


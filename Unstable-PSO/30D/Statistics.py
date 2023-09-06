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

def fct_concentration(X, Gl,  eps):
    Concentration=np.zeros(T_PSO_short)
    for t in range(T_PSO_short):
        nb=0
        for i in range(n):
            if np.linalg.norm(X[:,i,t]-Gl[:,t])<eps:
                nb=nb+1
    
        Concentration[t]=nb/n
    
    return Concentration

#%%

ConcentrationHarmonic=np.zeros(T_PSO_short)
ConcentrationOrbit=np.zeros(T_PSO_short)
ConcentrationClassic=np.zeros(T_PSO_short)
for s in range(sim):
    eps=1.3552186944893772
   

    XgOrbit=np.load(path+'G_s' + str(nameOrbit) + str(s) + '.npy')
    XgClassic=np.load(path+'G_s' + str(nameClassic) + str(s) + '.npy' )
    XgHarmonic=np.load(path+'G_s' + str(nameHarmonic) + str(s) + '.npy' )
    
    XOrbit=np.load(path+'X_s' + str(nameOrbit) + str(s) + '.npy')
    XClassic=np.load(path+'X_s' + str(nameClassic) + str(s) + '.npy')
    XHarmonic=np.load(path+'X_s' + str(nameHarmonic) + str(s) + '.npy')
    
    ConcentrationOrbit=ConcentrationOrbit+fct_concentration(XOrbit, XgOrbit,eps)
    ConcentrationClassic=ConcentrationClassic+fct_concentration(XClassic,XgClassic,  eps)
    ConcentrationHarmonic=ConcentrationHarmonic+fct_concentration(XHarmonic, XgHarmonic,eps)
    
np.save('./Results/AverageConcentrationRastrigin'+str(nameOrbit)+'.npy', ConcentrationOrbit/sim)
np.save('./Results/AverageConcentrationRastrigin'+str(nameClassic)+'.npy', ConcentrationClassic/sim)
np.save('./Results/AverageConcentrationRastrigin'+str(nameHarmonic)+'.npy', ConcentrationHarmonic/sim)
    
print('done concentration')

#%% Mean Orbit 
delta=1
T_PSO_short=1000
LossO=np.zeros(int(T_PSO_short/delta))

for t in range(int(T_PSO_short/delta)):
    
    List=[]
    for s in range(sim):
        OrbitG=np.load(path+'G_s'+str(nameOrbit)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(OrbitG[:,int(t*delta)])
        List.append(functionvalue)
        
    LossO[t]=np.mean(List)

np.save('./Results/ShortMeanRastrigin'+str(nameOrbit)+'', LossO)
print(len(LossO))

#%% Mean Harmonic
        
LossH=np.zeros(int(T_PSO_short/delta))
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        HarmonicG=np.load(path+'G_s'+str(nameHarmonic)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(HarmonicG[:,int(t*delta)])
        List.append(functionvalue)
        
    LossH[t]=np.mean(List)
   

np.save('./Results/ShortMeanRastrigin'+str(nameHarmonic)+'', LossH)

#%% Mean Classic
        
LossC=np.zeros(int(T_PSO_short/delta))
for t in range(int(T_PSO_short/delta)):
    List=[]
    for s in range(sim):
        ClassicG=np.load(path+'G_s'+str(nameClassic)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(ClassicG[:,int(t*delta)])
        List.append(functionvalue)
        
    LossC[t]=np.mean(List)
    
np.save('./Results/ShortMeanRastrigin'+str(nameClassic)+'', LossC)

print('done mean')

#%%
'''
T_PSO=500000
delta=10000
LossH=np.zeros((int(T_PSO/delta),sim))
for t in range(int(T_PSO/delta)):
    List=[]
    for s in range(sim):
        HarmonicG=np.load(path+'G_s'+str(nameOrbit)+str(s)+'.npy')
        functionvalue=fct_Rastrigin(HarmonicG[:,int(t*delta)])
       
        
        LossH[t,s]=functionvalue
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:24:30 2023

@author: bzfkostr
"""

import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import math
from multiprocessing import Pool
from Functions import *
from Variables import *
import sys

#%% multiprocessing

if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)

#%% PSO algorithm
    
def PSO(start, name, w, mu): # start = number of simulation, name = labelling for saving the file, w and mu are the parameters of PSO
    
    np.random.seed(int(start)) # set random seed
   
    # select start values

    X0=np.copy(StartWerte[0:dim,:,start])

    # initialize local and global best

    LocalBest=np.empty([dim, n,T_PSO]) # Local best positions of all particles for each timestep
    Loss=np.empty([n,T_PSO]) # functionvalue of local best postions

    GlobalBest=np.empty([ dim,T_PSO]) # globale best position for each timestep
    GlobalBestLoss=np.empty([T_PSO]) # functionvalue of global best position
    
    X=np.empty([dim, n,T_PSO]) # location of particles
    V=np.empty([dim, n,T_PSO]) # velocity of particles
    

    # initialize particles position X and velocity V
    X[:,:,0]=X0
    V0=np.zeros([dim,n])
    V[:,:,0]=V0

    # initialize local and global best
    for i in range(n):
        Loss[i,0]=fct_loss(X[:,i,0])
        LocalBest[:,i,0]=X[:,i,0]
        if i==0:
            GlobalBest[:,0]=X[:,i,0]
            GlobalBestLoss[0]=Loss[i,0]
        elif Loss[i,0]<GlobalBestLoss[0]:
            GlobalBest[:,0]=X[:,i,0]
            GlobalBestLoss[0]=Loss[i,0]
        else:
            pass

    # perform initial step

    for t in range(T_PSO-1):
        
        GlobalBest[:,t+1]=GlobalBest[:,t] # initialize global best position of the next step
        
        # calculate new velcoity and position of each particle
        for i in range(n):
            V[:,i,t+1]=w*V[:,i,t]+mu*np.random.uniform(0,1)*(LocalBest[:,i,t]-X[:,i,t])+mu*np.random.uniform(0,1)*(GlobalBest[:,t]-X[:,i,t])
            #V[:,i,t+1]=w*V[:,i,t]+mu*np.random.uniform(0,1,5)*(LocalBest[:,i,t]-X[:,i,t])+mu*np.random.uniform(0,1,5)*(GlobalBest[:,t]-X[:,i,t])
            X[:,i,t+1]=X[:,i,t]+V[:,i,t+1]
            
            # calculate the loss of the new position
            
            Loss[i,t+1]=fct_loss(X[:,i,t+1])
            
            # update local best
            if Loss[i,t+1]<fct_loss(LocalBest[:,i,t]):
                LocalBest[:,i,t+1]=X[:,i,t+1]
            else:
                LocalBest[:,i,t+1]=LocalBest[:,i,t]
                
        # update global best
        for i in range(n): 
            
            if Loss[i,t+1]<fct_loss(GlobalBest[:,t+1]):
                GlobalBest[:,t+1]=X[:,i,t+1]
                
        GlobalBestLoss[t+1]=fct_loss(GlobalBest[:,t+1])
        
        
    # save results
    
    np.save(path+'L_s'+str(name)+str(start)+'.npy', LocalBest)
    np.save(path+'X_s'+str(name)+str(start)+'.npy', X)
    np.save(path+'G_s'+str(name)+str(start)+'.npy', GlobalBest)


#%% run all PSO configurations

PSO(simulation, nameOrbit,w_o, mu_o)
PSO(simulation, nameHarmonic, w_h, mu_h)
PSO(simulation, nameClassic, w_c, mu_c)



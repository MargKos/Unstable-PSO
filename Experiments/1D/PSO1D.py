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
import sys
from Variables import *

#%% for multiprocessing
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)

#%%

# loop over all 1D test functions

for function_number in range(8):
    
    fct_loss=FunctionList[function_number]
    StartWerte=StartingPoints[:,:,:, function_number] 
    
    def PSO(start, name, w, mu): # start = number of simulation, name = name to save the results, w = inertia weight, mu = acceleration coefficient
        
        np.random.seed(int(start)) # set the seed for each simulation

        # select start values

        X0=np.copy(StartWerte[0:dim,:,start]) 
        
        # initialize local and global best and their loss

        LocalBest=np.empty([dim, n,T_PSO])
        Loss=np.empty([n,T_PSO])
        GlobalBest=np.empty([ dim,T_PSO])
        GlobalBestLoss=np.empty([T_PSO])
        
        # initialize position and velocity

        X=np.empty([dim, n,T_PSO])
        V=np.empty([dim, n,T_PSO])
        

        # initialize particles and velocity at time 0

        X[:,:,0]=X0
        V0=np.zeros([dim,n])
        V[:,:,0]=V0

        # calculate local and global best at time step 0
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

        # perform iterations

        for t in range(T_PSO-1):
            
            GlobalBest[:,t+1]=GlobalBest[:,t] # initialize global best position of the next step
            
            # calculate new velcoity and position of each particle
            
            for i in range(n):
                V[:,i,t+1]=w*V[:,i,t]+mu*np.random.uniform(0,1)*(LocalBest[:,i,t]-X[:,i,t])+mu*np.random.uniform(0,1)*(GlobalBest[:,t]-X[:,i,t])
                X[:,i,t+1]=X[:,i,t]+V[:,i,t+1]
                
                # calculate the loss of the new position
                
                Loss[i,t+1]=fct_loss(X[:,i,t+1])
                
                # update local best

                if Loss[i,t+1]<fct_loss(LocalBest[:,i,t]):
                    LocalBest[:,i,t+1]=X[:,i,t+1]
                else:
                    LocalBest[:,i,t+1]=LocalBest[:,i,t]
                    
            # update global best if better position was found

            for i in range(n): 
                
                if Loss[i,t+1]<fct_loss(GlobalBest[:,t+1]):
                    GlobalBest[:,t+1]=X[:,i,t+1]
                    
            GlobalBestLoss[t+1]=fct_loss(GlobalBest[:,t+1])
            
            
        # save results in the path defines aboth
        np.save(path+'L_s'+str(name)+str(start)+'.npy', LocalBest)
        np.save(path+'X_s'+str(name)+str(start)+'.npy', X)
        np.save(path+'G_s'+str(name)+str(start)+'.npy', GlobalBest)
       

    # run all three parameter combinations
    

    PSO(simulation, nameDivergent[function_number],w_divergent, mu_divergent)
    PSO(simulation, nameDamped[function_number], w_damped, mu_damped)
    PSO(simulation, nameOverdamped[function_number], w_overdamped, mu_overdamped)



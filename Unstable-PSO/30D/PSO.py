#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:44:31 2023

@author: bzfkostr
"""

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from multiprocessing import Pool
from Rastrigin_fct import *       
import sys
from Variables import *

# Print task number
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)



#%% define PSO

def PSO(start, name, w, mu, counter): # name: how to save the file, w: weight, mu: learning rate, counter: number of iterations a particle has to stay in the same local minimum to be saved, start: number of the start value
    np.random.seed(int(start)) # set seed for reproducibility

    X0 = StartWerte[0:dim, :, start] # start values for each particle

    # create an array will global best positions of the swarm of each time-step

    Gbest=np.zeros((dim, T_PSO)) # dim: dimension of the problem, T_PSO: number of time-steps

    # find the best X0 value for each particle and the whole swarm 

    g_best=X0[:,0]
    for i in range(n-1):
        if fct_Rastrigin(X0[:,i+1])<fct_Rastrigin(g_best):
            g_best=X0[:,i+1]
    
    Gbest[:,0]=g_best

    # calculate the fct value of each particle of X0

    fct_value=np.zeros(n)

    for i in range(n):
        fct_value[i]=fct_Rastrigin(X0[:,i])

    v0=np.zeros((dim,n))

    # give the local best position of each particle

    p_best=X0 

    # save local minima

    Minima=[] # list of local minima
    TimesMinima=[] # list of time-steps when local minima were found

    # do the time-iteration of PSO T_PSO-1 times
    
    count=np.zeros(n) # counter with the number of iterations it did not change the local best position

    
    # create for each particle an empty list that will be filled later with the number of iterations it did not change the local best position

    NumberofIterations=[None]*n 

    for i in range(n):
        NumberofIterations[i]=[]
        
    for t in range(T_PSO-1):
        
        # calculate the velocity of each particle

        for i in range(n):
            v0[:,i]=w*v0[:,i]+mu*np.random.rand()*(p_best[:,i]-X0[:,i])+mu*np.random.rand()*(g_best-X0[:,i])
        
        # calculate the position of each particle

        X0=X0+v0
        

        # calculate the fct value of each particle
        
        for i in range(n):
            fct_value[i]=fct_Rastrigin(X0[:,i])

        # update the local best position of each particle

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(p_best[:,i]):
                p_best[:,i]=X0[:,i] # update the local best position
                NumberofIterations[i].append(count[i])
                if count[i]>counter: # if the particle did not change its local best position for counter iterations, save it as a local minimum
                    Minima.append(X0[:,i])   # save the local minimum
                    TimesMinima.append(t)    # save the time-step when the local minimum was found
                count[i]=0 # reset the counter
            else:
                count[i]=count[i]+1 # increase the counter by 1

        # update the global best position of the swarm

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(g_best):
                g_best=X0[:,i]
        
        Gbest[:,t+1]=g_best
    
    np.save(path+'G_s' + str(name) + str(start) + '.npy',  Gbest) # save the global best position of the swarm
    np.save(path+'Minima_s' + str(name) + str(start) + '.npy',  Minima) #   save the local minima
    np.save(path+'TimesMinima_s' + str(name) + str(start) + '.npy',  TimesMinima) # save the time-steps when the local minima were found
    np.save(path+'NumberofIterations_s' + str(name) + str(start) + '.npy',  NumberofIterations) # save the time-steps when the local minima were found
    'print done PSO'
    return 

#%%

# select parameters for PSO

w_o, mu_o=Orbit()
PSO(simulation, nameOrbit,w_o, mu_o, counter_Orbit)


w_h, mu_h=Harmonic()
PSO(simulation, nameHarmonic, w_h, mu_h, counter_Harmonic)


w_c, mu_c=Classic()
PSO(simulation, nameClassic, w_c, mu_c, counter_Classic)

    
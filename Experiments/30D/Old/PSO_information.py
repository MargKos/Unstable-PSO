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
from Variables import *   
import sys

'''Short PSO which saves the position of all particles to determine the waiting times '''

# Print task number
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)



# define PSO

def PSO_s(start, name, w, mu): # start is the number of the starting point, name is the name of the simulation, w and mu are the parameters of PSO
    np.random.seed(int(start)) # set seed for each starting point

    # set start values
    X0 = np.copy(StartWerte[0:dim, :, start])
    
    # create an array will global best positions of the swarm of each time-step

    Gbest=np.zeros((dim, T_PSO_short))

    # find the best X0 value for each particle and the whole swarm

    g_best=X0[:,0]
    for i in range(n-1):
        if fct_Rastrigin(X0[:,i+1])<fct_Rastrigin(g_best):
            g_best=X0[:,i+1]

    # initialize global best position of the swarm

    Gbest[:,0]=g_best

    # calculate the fct value of each particle of X0

    fct_value=np.zeros(n)

    for i in range(n):
        fct_value[i]=fct_Rastrigin(X0[:,i])

    # initialize velocity of each particle

    v0=np.zeros((dim,n))

    # give the local best position of each particle

    p_best=X0

    # create for each particle an empty list that will be filled later with the number of iterations it did not change the local best position
  
    ParticlesPositions=np.empty([dim, n,T_PSO_short]) # only for short runs

    ParticlesPositions[:,:,0]=X0
    for t in range(T_PSO_short-1):
        
        # calculate the velocity of each particle

        for i in range(n):
            v0[:,i]=w*v0[:,i]+mu*np.random.rand()*(p_best[:,i]-X0[:,i])+mu*np.random.rand()*(g_best-X0[:,i])
            #v0[:,i]=w*v0[:,i]+mu*np.random.uniform(0,1,30)*(p_best[:,i]-X0[:,i])+mu*np.random.uniform(0,1,30)*(g_best-X0[:,i])
        
        # calculate the position of each particle

        X0=X0+v0

        # save the position of each particle
        ParticlesPositions[:,:,t+1]=X0

        # calculate the fct value of each particle
        
        for i in range(n):
            fct_value[i]=fct_Rastrigin(X0[:,i])

        # update the local best position of each particle

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(p_best[:,i]):
                p_best[:,i]=X0[:,i]
               

        # update the global best position of the swarm

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(g_best):
                g_best=X0[:,i]
        
        Gbest[:,t+1]=g_best
    
    # save particles positions
    np.save(path+'X_s' + str(name) + str(start) + '.npy', ParticlesPositions)
    
    return Gbest, ParticlesPositions

#%% run PSO for each simulation

Gbest_Divergent,X_Divergent=PSO_s(simulation, nameDivergent,w_divergent, mu_divergent)
Gbest_Damped,X_Damped=PSO_s(simulation, nameDamped, w_damped, mu_damped)
G_Overdamped,X_Overdamped=PSO_s(simulation, nameOverdamped, w_overdamped, mu_overdamped)
    
     

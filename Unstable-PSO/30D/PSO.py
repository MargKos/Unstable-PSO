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


# Print task number
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)


# define PSO

def PSO(start, name, w, mu, counter): # start is the number of the starting point, name is the name of the simulation, w and mu are the parameters of PSO, counter is the waiting time to locate the local minima

    np.random.seed(int(start)) # set seed for each starting point

    # set start values

    X0 = np.copy(StartWerte[0:dim, :, start])

    # create an array will global best positions of the swarm of each time-step

    Gbest=np.zeros((dim, T_PSO))

    # find the best X0 value for each particle and the whole swarm

    g_best=X0[:,0]
    for i in range(n-1):
        if fct_Rastrigin(X0[:,i+1])<fct_Rastrigin(g_best):
            g_best=X0[:,i+1]
    
    Gbest[:,0]=g_best

    # calculate the fct value of each particle of X0, to determine the local best position of each particle in the next step

    fct_value=np.zeros(n)

    for i in range(n):
        fct_value[i]=fct_Rastrigin(X0[:,i])

    v0=np.zeros((dim,n))

    # give the local best position of each particle

    p_best=X0

    # save local minima and the times they were found

    Minima=[] # list of local minima
    TimesMinima=[] # list of times the local minima were found

    count=np.zeros(n) # counts for each particle the number of iterations it did not change the local best position
    
    # create for each particle an empty list that will be filled later with the number of iterations it did not change the local best position
    # for statistics about waiting times
    NumberofIterations=[None]*n

    for i in range(n):
        NumberofIterations[i]=[]
        
    for t in range(T_PSO-1):
        
        # calculate the velocity of each particle

        for i in range(n):
            v0[:,i]=w*v0[:,i]+mu*np.random.rand()*(p_best[:,i]-X0[:,i])+mu*np.random.rand()*(g_best-X0[:,i])
            #v0[:,i]=w*v0[:,i]+mu*np.random.uniform(0,1,30)*(p_best[:,i]-X0[:,i])+mu*np.random.uniform(0,1,30)*(g_best-X0[:,i])
        
        # calculate the position of each particle

        X0=X0+v0
        
        # calculate the fct value of each particle
        
        for i in range(n):
            fct_value[i]=fct_Rastrigin(X0[:,i])

        # update the local best position of each particle

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(p_best[:,i]):
                p_best[:,i]=X0[:,i]
                # save the number of iterations it did not change the local best position
                NumberofIterations[i].append(count[i])
                # if the number of iterations it did not change the local best position is greater than the waiting time to locate the local minima, save the local minima and the time it was found
                if count[i]>counter:
                    Minima.append(X0[:,i])  
                    TimesMinima.append(t)
                # reset the counter
                count[i]=0
            else:
                # if the local best position did not change, increase the counter
                count[i]=count[i]+1

        # update the global best position of the swarm

        for i in range(n):
            if fct_value[i]<fct_Rastrigin(g_best):
                g_best=X0[:,i]
        
        Gbest[:,t+1]=g_best
    # save the results
    np.save(path+'G_s' + str(name) + str(start) + '.npy', Gbest)
    np.save(path+'Minima_s' + str(name) + str(start) + '.npy', Minima)
    np.save(path+'NumberofIterations_s' + str(name) + str(start) + '.npy', NumberofIterations)
    np.save(path+'TimesMinima_s' + str(name) + str(start) + '.npy', TimesMinima)
    
    return Gbest, NumberofIterations, Minima

#%%

# run PSO for each configuration
GbestH, itH,MH=PSO(simulation, nameHarmonic, w_h, mu_h, counter_Harmonic)

GbestO, itO,MO=PSO(simulation, nameOrbit,w_o, mu_o, counter_Orbit)

GbestC, itC,MC=PSO(simulation, nameClassic, w_c, mu_c, counter_Classic)

 
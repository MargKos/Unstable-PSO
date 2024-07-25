#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:33:42 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Rastrigin_fct import *
# %% sampling Range

b1=-5.12
b2=5.12



#%% Check if loccal Minima

def gradient(x): # function that calculates the gradient at x
    
    grad=np.zeros(dim)
    
    for i in range(dim):
        grad[i]=2*(x[i]+np.pi*10*np.sin(2*np.pi*x[i]))
    
    return grad

def hesse(x): # calculates the Hessian matrix and says if local minimum or local maximum 
    
    Hesse=np.zeros((dim, dim))
    
    for i in range(dim):
       
        Hesse[i,i]=2*(1+20*(np.pi**2)*np.cos(2*np.pi*x[i]))
    
    if np.all(np.linalg.eigvals(Hesse) > 0):
     
        return 'min'
    
    if np.all(np.linalg.eigvals(Hesse) < 0):
        
        return 'max'


def find_localminima(start):
    res = minimize(fct_Rastrigin, start, method='CG',options={'disp': False, 'maxiter':1000000})

    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            return sol
        
        else:
            return 'not'

#%% define random search

dg=0.000005
dl=0.5

def random_search_(number_samples, T, bd1, bd2,  dim):
    
    NumberOfFoundLocalMinima=np.zeros(T)
   
    LocalMinima=[]
    for t in range(T):
        
        # generate number_samples samples
        
        Samples=[]
        for i in range(number_samples):
            x=np.random.uniform(-5.12, -2, dim)
            y=np.random.uniform(2, 5.12, dim)
            
            x=np.concatenate((x,y))
            
            Samples.append(x)
        
        
        for  i in range(number_samples):
             res = minimize(fct_Rastrigin, Samples[i], method='CG',options={'disp': False, 'maxiter':1000000})

             sol=res.x
             
             if np.max(np.abs(gradient(sol)))<dg and hesse(sol)=='min':
                 
                 if len(LocalMinima)==0 and fct_Rastrigin(sol)<353: # accept only solutions with values better than the value of the average minimum found by PSO
                        
                     LocalMinima.append(sol)
                
                 # test if new local minima
                 else:
                     count=0
                     if len(LocalMinima)>0:
                         for j in range(len(LocalMinima)):
                             if np.linalg.norm(sol-LocalMinima[j])<dl:
                                count=1
                                print(j)
                                break
                        
                         if j==len(LocalMinima)-1 and fct_Rastrigin(sol)<353:
                            LocalMinima.append(sol)
                    
                    
                                  
               
        NumberOfFoundLocalMinima[t]=len(LocalMinima)
        print(len(LocalMinima), t)
       
    return NumberOfFoundLocalMinima, LocalMinima
            
            
        
        
#%% 

number_samples=20
T=58
simulations=100
dim=30
NumberOfFoundLocalMinima=np.zeros(T)
#%% Rastrigin

for i in range(simulations):
    NumberOfFoundLocalMinima_s, LocalMinima=random_search_(number_samples, T,b1,b2, dim)
    NumberOfFoundLocalMinima=NumberOfFoundLocalMinima+NumberOfFoundLocalMinima_s
    np.save('./Results/RandomSamplingLocalMinima'+str(i)+'.npy', LocalMinima)

NumberOfFoundLocalMinima=NumberOfFoundLocalMinima/simulations

np.save('./Results/RandomSamplingshortsim57.npy', NumberOfFoundLocalMinima)

#%%
#NumberOfFoundLocalMinima=np.load('./Results/RandomSampling2.npy')
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinima)
plt.savefig('RS2.png')

stop
#%% Check gradient of minima from PSO


AllLocalMinimaOverdamped=np.load('./Results/AllLocalMinima'+str(nameOverdamped)+'.npy')
AllLocalMinimaDamped=np.load('./Results/AllLocalMinima'+str(nameDamped)+'.npy')
AllLocalMinimaDivergent=np.load('./Results/AllLocalMinima'+str(nameDivergent)+'.npy')


Grad1=[]

for i in range(len(AllLocalMinimaDamped)):
    Grad1.append(np.max(np.abs(gradient(AllLocalMinimaDamped[i]))))
    print(np.linalg.norm(AllLocalMinimaDamped[i]-AllLocalMinimaDamped[1]))


Grad2=[]
for i in range(len(AllLocalMinimaOverdamped)):
    Grad2.append(np.max(np.abs(gradient(AllLocalMinimaOverdamped[i]))))

Grad3=[]
for i in range(len(AllLocalMinimaDivergent)):
    Grad3.append(np.max(np.abs(gradient(AllLocalMinimaDivergent[i]))))

#%% Function to calculate average distance
from scipy.spatial.distance import pdist, squareform
def minimal_distance(local_minima):
    if len(local_minima) <= 1:
        return 0
    dist_matrix = pdist(local_minima, 'euclidean')
    return np.mean(dist_matrix)

# Calculate minimal distances for each array
min_distance_overdamped = minimal_distance(AllLocalMinimaOverdamped)
min_distance_damped = minimal_distance(AllLocalMinimaDamped)
min_distance_divergent = minimal_distance(AllLocalMinimaDivergent)

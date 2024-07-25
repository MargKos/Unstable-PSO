# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:41:07 2024

@author: mkost
"""

import numpy as np
import matplotlib.pyplot as plt
from Rastrigin_fct import *
from scipy.optimize import minimize
import sys
from Variables import *

    
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

simulation=0
#%% for all minima
    
TimeLocalMinimaDivergent=[] # list of times when local minima was found
LocalMinimaDivergent=[]     # list of local minima
DistanceDivergent=[]        # list of distances from local minima to staedy local best positions
NumberOfFoundLocalMinimaDivergent=np.zeros(T_PSO) # gives the total number of local minima found at time t

# load steady local bst positions and their times from one single simulation 
TimeMinimaDivergent_s=np.load(path+'TimesMinima_s' + str(nameDivergent) + str(simulation) + '.npy')  
MinimaDivergent=np.load(path+'Minima_s' + str(nameDivergent) + str(simulation) + '.npy')

'''
 steady local best positions can be the same for different particles, threrefore we 
 create groups of elements from loaded MinimaDivergent and save them in GroupedMinimaDivergent such that all elements in one group are closer than 0.1 to save comput. time
'''
GroupedMinimaDivergent=[] # List of groups of minima 
GroupedTimeMinimaDivergent=[] # List of the corresponding times


for i in range(len(MinimaDivergent)): # go through all minima of one simulation
    if i==0: # if first element create a new group
        GroupedMinimaDivergent.append([])
        GroupedTimeMinimaDivergent.append([])
        
        GroupedMinimaDivergent[0].append(MinimaDivergent[i])
        GroupedTimeMinimaDivergent[0].append(TimeMinimaDivergent_s[i])
    else:
        # find the right group for MinimaDivergent[i]
        for j in range(len(GroupedMinimaDivergent)): # go though all groups
            for k in range(len(GroupedMinimaDivergent[j])): # go through all elements in one group
                if np.linalg.norm(MinimaDivergent[i]-GroupedMinimaDivergent[j][k])<0.1: # if MinimaDivergent[i] is closer than 0.1 to one element in the group j, then add this element
                    GroupedMinimaDivergent[j].append(MinimaDivergent[i]) # add Minima[i] to the group
                    GroupedTimeMinimaDivergent[j].append(TimeMinimaDivergent_s[i]) # add the corresponding time to the group
                    # break the j and k loop, and go to the next minimum, as this element was classified
                    break
            else:
                continue
            break
        # if no group found, create a new group
        if j==len(GroupedMinimaDivergent)-1:
            GroupedMinimaDivergent.append([])
            GroupedTimeMinimaDivergent.append([])
            GroupedMinimaDivergent[j+1].append(MinimaDivergent[i])
            GroupedTimeMinimaDivergent[j+1].append(TimeMinimaDivergent_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

MinimaDivergent=[]
TimeMinimaDivergent_s=[]
for i in range(len(GroupedMinimaDivergent)):

    MinimaDivergent.append(np.mean(GroupedMinimaDivergent[i], axis=0))
    TimeMinimaDivergent_s.append(np.min(GroupedTimeMinimaDivergent[i]))

# MinimaDivergent gives now a list of unique steady local best positions


#%% count number of found local minima found at time t w.r.t to TimeMinimaDivergent_s

NumberOfFoundLocalMinimaDivergent=np.zeros(T_PSO) 

foundminima=0
for t in range(T_PSO):
    for i in range(len(TimeMinimaDivergent_s)):
        if TimeMinimaDivergent_s[i]==t:
            foundminima=foundminima+1
    
    NumberOfFoundLocalMinimaDivergent[t]=foundminima

# plot the number of found local minima w.r.t. TimeMinimaDivergent_s
#%%
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaDivergent[0:3000])

plt.xlabel('Time')
plt.ylabel('Number of Steady Local Best Positions')


#%% same for damped parameters, for details see the code above

AverageNumberOfFoundLocalMinimaDamped=np.zeros(T_PSO)

TimeLocalMinimaDamped=[]
LocalMinimaDamped=[]
DistanceDamped=[]
NumberOfFoundLocalMinimaDamped=np.zeros(T_PSO)

TimeMinimaDamping_s=np.load(path+'TimesMinima_s' + str(nameDamped) + str(simulation) + '.npy')  
MinimaDamped=np.load(path+'Minima_s' + str(nameDamped) + str(simulation) + '.npy')

# create groups of elements from MinimaDamped such that all elements in one group are closer than 0.1

GroupedMinimaDamped=[]
GroupedTimeMinimaDamped=[]

for i in range(len(MinimaDamped)):
    print(i)
    if i==0:
        GroupedMinimaDamped.append([])
        GroupedTimeMinimaDamped.append([])
        
        GroupedMinimaDamped[0].append(MinimaDamped[i])
        GroupedTimeMinimaDamped[0].append(TimeMinimaDamping_s[i])
    else:
        # find the right group for MinimaDamped[i]
        for j in range(len(GroupedMinimaDamped)):
            for k in range(len(GroupedMinimaDamped[j])):
                if np.linalg.norm(MinimaDamped[i]-GroupedMinimaDamped[j][k])<0.1:
                    GroupedMinimaDamped[j].append(MinimaDamped[i])
                    GroupedTimeMinimaDamped[j].append(TimeMinimaDamping_s[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedMinimaDamped)-1:
            GroupedMinimaDamped.append([])
            GroupedTimeMinimaDamped.append([])
            GroupedMinimaDamped[j+1].append(MinimaDamped[i])
            GroupedTimeMinimaDamped[j+1].append(TimeMinimaDamping_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

MinimaDamped=[]
TimeMinimaDamping_s=[]
for i in range(len(GroupedMinimaDamped)):

    MinimaDamped.append(np.mean(GroupedMinimaDamped[i], axis=0))
    TimeMinimaDamping_s.append(np.min(GroupedTimeMinimaDamped[i]))

bb
#%% count 

NumberOfFoundLocalMinimaDamped=np.zeros(T_PSO) 

foundminima=0
for t in range(T_PSO):
    for i in range(len(TimeMinimaDamped_s)):
        if TimeMinimaDamped_s[i]==t:
            foundminima=foundminima+1
    
    NumberOfFoundLocalMinimaDamped[t]=foundminima

# plot the number of found local minima w.r.t. TimeMinimaDivergent_s
#%%
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaDamped[0:3000])

plt.xlabel('Time')
plt.ylabel('Number of Steady Local Best Positions')

#%%

AverageNumberOfFoundLocalMinimaOverdamped=np.zeros(T_PSO)

TimeLocalMinimaOverdamped=[]
LocalMinimaOverdamped=[]
DistanceOverdamped=[]
NumberOfFoundLocalMinimaOverdamped=np.zeros(T_PSO)

TimeMinimaOverdamped_s=np.load(path+'TimesMinima_s' + str(nameOverdamped) + str(simulation) + '.npy')  
MinimaOverdamped=np.load(path+'Minima_s' + str(nameOverdamped) + str(simulation) + '.npy')

# create groups of elements from Minima such that all elements in one group are closer than 0.1

GroupedMinimaOverdamped=[]
GroupedTimeMinimaOverdamped=[]

for i in range(len(MinimaOverdamped)):
    if i==0:
        GroupedMinimaOverdamped.append([])
        GroupedTimeMinimaOverdamped.append([])
        
        GroupedMinimaOverdamped[0].append(MinimaOverdamped[i])
        GroupedTimeMinimaOverdamped[0].append(TimeMinimaOverdamped_s[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedMinimaOverdamped)):
            for k in range(len(GroupedMinimaOverdamped[j])):
                if np.linalg.norm(MinimaOverdamped[i]-GroupedMinimaOverdamped[j][k])<0.1:
                    GroupedMinimaOverdamped[j].append(MinimaOverdamped[i])
                    GroupedTimeMinimaOverdamped[j].append(TimeMinimaOverdamped_s[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedMinimaOverdamped)-1:
            GroupedMinimaOverdamped.append([])
            GroupedTimeMinimaOverdamped.append([])
            GroupedMinimaOverdamped[j+1].append(MinimaOverdamped[i])
            GroupedTimeMinimaOverdamped[j+1].append(TimeMinimaOverdamped_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

MinimaOverdamped=[]
TimeMinimaOverdamped_s=[]
for i in range(len(GroupedMinimaOverdamped)):

    MinimaOverdamped.append(np.mean(GroupedMinimaOverdamped[i], axis=0))
    TimeMinimaOverdamped_s.append(np.min(GroupedTimeMinimaOverdamped[i]))

#%% count 

NumberOfFoundLocalMinimaOverdamped=np.zeros(T_PSO) 

foundminima=0
for t in range(T_PSO):
    for i in range(len(TimeMinimaOverdamped_s)):
        if TimeMinimaOverdamped_s[i]==t:
            foundminima=foundminima+1
    
    NumberOfFoundLocalMinimaOverdamped[t]=foundminima

# plot the number of found local minima w.r.t. TimeMinimaDivergent_s
#%%
fig=plt.figure()
plt.plot(NumberOfFoundLocalMinimaOverdamped[0:3000])

plt.xlabel('Time')
plt.ylabel('Number of Steady Local Best Positions')

#%% Run Random Sampling from x points where x is the number of found local minima at time t

RS_divergent=np.zeros(np.max(TimeMinimaDivergent_s))
# Divergent
counter=0
number_of_found_minima=0
for t in range(len(TimeMinimaDivergent_s)):

    number=int(NumberOfFoundLocalMinimaDivergent[t])

    # sample in number uniformly distributed points in [-5.12,-2] times [2,5.12] in 30 dimensions
    
    for i in range(number):
        
            
        x=np.random.uniform(-5.12, 0, dim)
        y=np.random.uniform(0, 5.12, dim)
        counter=counter+1
        x=np.concatenate((x,y))
        
        # run CG
        
        res = minimize(fct_Rastrigin, MinimaDamped[i], method='CG',options={'disp': False, 'maxiter':1000000})

        sol=res.x
        
        # chech if x local minimum

        if np.max(np.abs(gradient(sol)))<0.01 and  hesse(sol)=='min':
            number_of_found_minima=number_of_found_minima+1
          
    RS_divergent[t]=number_of_found_minima
            
fig=plt.figure()
plt.plot(RS_divergent)
#%%
# Overdamped

RS_overdamped=np.zeros(len(TimeMinimaOverdamped_s))

for t in range(len(TimeMinimaOverdamped_s)):
        
    number=NumberOfFoundLocalMinimaOverdamped[t]

    # sample in number uniformly distributed points in [-5.12,-2] times [2,5.12] in 30 dimensions

    for i in range(number):
            
        x=np.random.uniform(-5.12, -2, dim)
        y=np.random.uniform(2, 5.12, dim)
        
        x=np.concatenate((x,y))
        
        # chech if x local minimum

        if np.max(np.abs(gradient(x)))<0.01 and  hesse(x)=='min':
        
            RS_overdamped[t]=RS_overdamped[t]+1

# Damped


RS_damped=np.zeros(len(TimeMinimaDamped_s))

for t in range(len(TimeMinimaDamped_s)):
            
    number=NumberOfFoundLocalMinimaDamped[t]

    # sample in number uniformly distributed points in [-5.12,-2] times [2,5.12] in 30 dimensions

    for i in range(number):
            
        x=np.random.uniform(-5.12, -2, dim)
        y=np.random.uniform(2, 5.12, dim)
        
        x=np.concatenate((x,y))
        
        # chech if x local minimum

        if np.max(np.abs(gradient(x)))<0.01 and  hesse(x)=='min':
        
            RS_damped[t]=RS_damped[t]+1
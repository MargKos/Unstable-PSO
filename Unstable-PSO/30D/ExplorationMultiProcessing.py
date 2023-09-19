import numpy as np
import matplotlib.pyplot as plt
from Rastrigin_fct import *
from scipy.optimize import minimize
from sklearn.manifold import TSNE
import sys
from Variables import *



# Print task number
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task number', simulation)


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

# Run Conjugate Gradient (CG) and use the unique elemnts of MinimaDivergent as staring points
for i in range(len(MinimaDivergent)): # go through all unique steady local best positions
    res = minimize(fct_Rastrigin, MinimaDivergent[i], method='CG',options={'disp': False, 'maxiter':1000000})
    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01: # check if gradient is small minima  
        if hesse(sol)=='min': # check if min, max or saddlepoint  
            LocalMinimaDivergent.append(sol) # append local minima to list of localminima
            DistanceDivergent.append(np.linalg.norm(MinimaDivergent[i]-sol)) # save distance from steady local best position to local minima

# now we can calculate the element of NumberOfFoundLocalMinimaO from TimeLocalMinimaO 

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaDivergent)):
        if TimeLocalMinimaDivergent[i]<t:
            NumberOfFoundLocalMinimaDivergent[t]+=1
            
# save all results: TimeLocalMinimaO, LocalMinimaO, DistanceO, NumberOfFoundLocalMinimaO

np.save(path+'NumberOfFoundLocalMinima'+str(nameDivergent)+str(simulation)+'', NumberOfFoundLocalMinimaDivergent)
np.save(path+'Distance'+str(nameDivergent)+str(simulation)+'', DistanceDivergent)
np.save(path+'LocalMinima'+str(nameDivergent)+str(simulation)+'', LocalMinimaDivergent)
np.save(path+'TimeLocalMinima'+str(nameDivergent)+str(simulation)+'', TimeLocalMinimaDivergent)

print('done exploration divergent parameters')
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


# Run CG and use the elemnts of Minima as staring points
for i in range(len(MinimaDamped)):
        
    res = minimize(fct_Rastrigin, MinimaDamped[i], method='CG',options={'disp': False, 'maxiter':1000000})

    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            LocalMinimaDamped.append(sol)
            DistanceDamped.append(np.linalg.norm(MinimaDamped[i]-sol))
            TimeLocalMinimaDamped.append(TimeMinimaDamping_s[i])

# group all elements from LocalMinimumD in group such that all ellemnts in one group are closer than 0.1 

GroupedLocalMinimaDamped=[]
GroupedTimeLocalMinimaDamped=[]
GroupedDistanceDamped=[]

for i in range(len(LocalMinimaDamped)):
    if i==0:
        GroupedLocalMinimaDamped.append([])
        GroupedTimeLocalMinimaDamped.append([])
        GroupedDistanceDamped.append([])
        
        GroupedLocalMinimaDamped[0].append(LocalMinimaDamped[i])
        GroupedTimeLocalMinimaDamped[0].append(TimeLocalMinimaDamped[i])
        GroupedDistanceDamped[0].append(DistanceDamped[i])
    else:
        # find the right group for MinimaDamped[i]
        for j in range(len(GroupedLocalMinimaDamped)):
            for k in range(len(GroupedLocalMinimaDamped[j])):
                if np.linalg.norm(LocalMinimaDamped[i]-GroupedLocalMinimaDamped[j][k])<0.1:
                    GroupedLocalMinimaDamped[j].append(LocalMinimaDamped[i])
                    GroupedTimeLocalMinimaDamped[j].append(TimeLocalMinimaDamped[i])
                    GroupedDistanceDamped[j].append(DistanceDamped[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedLocalMinimaDamped)-1:
            GroupedLocalMinimaDamped.append([])
            GroupedTimeLocalMinimaDamped.append([])
            GroupedDistanceDamped.append([])

            GroupedLocalMinimaDamped[j+1].append(LocalMinimaDamped[i])
            GroupedTimeLocalMinimaDamped[j+1].append(TimeMinimaDamping_s[i])
            GroupedDistanceDamped[j+1].append(DistanceDamped[i])
    

# create a new list of LocalMinimaDamped and TimeLocalMinimaDamped where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

LocalMinimaDamped=[]
TimeLocalMinimaDamped=[]
DistanceD=[]
for i in range(len(GroupedLocalMinimaDamped)):
    LocalMinimaDamped.append(np.mean(GroupedLocalMinimaDamped[i], axis=0))
    TimeLocalMinimaDamped.append(np.min(GroupedTimeLocalMinimaDamped[i]))
    DistanceDamped.append(np.mean(GroupedDistanceDamped[i]))

# now we can calculate the element of NumberOfFoundLocalMinimaD from TimeLocalMinimaD

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaDamped)):
        if TimeLocalMinimaDamped[i]<t:
            NumberOfFoundLocalMinimaDamped[t]+=1
            
# save results
       
np.save(path+'TimeLocalMinima'+str(nameDamped)+str(simulation)+'', TimeLocalMinimaDamped)
np.save(path+'Distance'+str(nameDamped)+str(simulation)+'', DistanceDamped)
np.save(path+'LocalMinima'+str(nameDamped)+str(simulation)+'', LocalMinimaDamped)
np.save(path+'NumberOfFoundLocalMinima'+str(nameDamped)+str(simulation)+'', NumberOfFoundLocalMinimaDamped)

print('done exploration damped parameters')

#%% Calculate LocalMinima, their times, distance for Overdamped parameters

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


# Run CG and use the elemnts of Minima as staring points
for i in range(len(MinimaOverdamped)):
    

    res = minimize(fct_Rastrigin, MinimaOverdamped[i], method='CG',options={'disp': False, 'maxiter':1000000})

    
    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            LocalMinimaOverdamped.append(sol)
            DistanceOverdamped.append(np.linalg.norm(MinimaOverdamped[i]-sol))
            TimeLocalMinimaOverdamped.append(TimeMinimaOverdamped_s[i])

# group all elements from LocalMinimumO in group such that all ellemnts in one group are closer than 0.1 

GroupedLocalMinimaOverdamped=[]
GroupedTimeLocalMinimaOverdamped=[]
GroupedDistanceOverdamped=[]

for i in range(len(LocalMinimaOverdamped)):
    if i==0:
        GroupedLocalMinimaOverdamped.append([])
        GroupedTimeLocalMinimaOverdamped.append([])
        GroupedDistanceOverdamped.append([])
        
        GroupedLocalMinimaOverdamped[0].append(LocalMinimaOverdamped[i])
        GroupedTimeLocalMinimaOverdamped[0].append(TimeLocalMinimaOverdamped[i])
        GroupedDistanceOverdamped[0].append(DistanceOverdamped[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedLocalMinimaOverdamped)):
            for k in range(len(GroupedLocalMinimaOverdamped[j])):
                if np.linalg.norm(LocalMinimaOverdamped[i]-GroupedLocalMinimaOverdamped[j][k])<0.1:
                    GroupedLocalMinimaOverdamped[j].append(LocalMinimaOverdamped[i])
                    GroupedTimeLocalMinimaOverdamped[j].append(TimeLocalMinimaOverdamped[i])
                    GroupedDistanceOverdamped[j].append(DistanceOverdamped[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedLocalMinimaOverdamped)-1:
            GroupedLocalMinimaOverdamped.append([])
            GroupedTimeLocalMinimaOverdamped.append([])
            GroupedDistanceOverdamped.append([])

            GroupedLocalMinimaOverdamped[j+1].append(LocalMinimaOverdamped[i])
            GroupedTimeLocalMinimaOverdamped[j+1].append(TimeMinimaOverdamped_s[i])
            GroupedDistanceOverdamped[j+1].append(DistanceOverdamped[i])
    

# create a new list of LocalMinimaO and TimeLocalMinimaO where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time
LocalMinimaOverdamped=[]
TimeLocalMinimaOverdamped=[]
DistanceC=[]
for i in range(len(GroupedLocalMinimaOverdamped)):
    LocalMinimaOverdamped.append(np.mean(GroupedLocalMinimaOverdamped[i], axis=0))
    TimeLocalMinimaOverdamped.append(np.min(GroupedTimeLocalMinimaOverdamped[i]))
    DistanceOverdamped.append(np.mean(GroupedDistanceOverdamped[i]))



# now we can calculate the element of NumberOfFoundLocalMinimaC from TimeLocalMinimaC 

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaOverdamped)):
        if TimeLocalMinimaOverdamped[i]<t:
            NumberOfFoundLocalMinimaOverdamped[t]+=1

    
np.save(path+'Distance'+str(nameOverdamped)+str(simulation)+'', DistanceOverdamped)
np.save(path+'NumberOfFoundLocalMinima'+str(nameOverdamped)+str(simulation)+'', NumberOfFoundLocalMinimaOverdamped)
np.save(path+'TimeLocalMinima'+str(nameOverdamped)+str(simulation)+'', TimeLocalMinimaOverdamped)
np.save(path+'LocalMinima'+str(nameOverdamped)+str(simulation)+'', LocalMinimaOverdamped)

print('done exploration overdamped parameters')
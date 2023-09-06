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

def gradient(x):
    
    grad=np.zeros(dim)
    
    for i in range(dim):
        grad[i]=2*(x[i]+np.pi*10*np.sin(2*np.pi*x[i]))
    
    return grad

def hesse(x):
    
    Hesse=np.zeros((dim, dim))
    
    for i in range(dim):
       
        Hesse[i,i]=2*(1+20*(np.pi**2)*np.cos(2*np.pi*x[i]))
    
    if np.all(np.linalg.eigvals(Hesse) > 0):
     #return H,'min'
        
        return 'min'
    
    if np.all(np.linalg.eigvals(Hesse) < 0):
        #return H, 'max'
        
        return 'max'

#%% for all minima
    

TimeLocalMinimaO=[]
LocalMinimaO=[]
DistanceO=[]
NumberOfFoundLocalMinima=np.zeros(T_PSO)

TimeMinimaOrbit_s=np.load(path+'TimesMinima_s' + str(nameOrbit) + str(simulation) + '.npy')  
Minima=np.load(path+'Minima_s' + str(nameOrbit) + str(simulation) + '.npy')

# create groups of elements from Minima such that all elements in one group are closer than 0.1

GroupedMinima=[]
GroupedTimeMinima=[]

for i in range(len(Minima)):
    if i==0:
        GroupedMinima.append([])
        GroupedTimeMinima.append([])
        
        GroupedMinima[0].append(Minima[i])
        GroupedTimeMinima[0].append(TimeMinimaOrbit_s[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedMinima)):
            for k in range(len(GroupedMinima[j])):
                if np.linalg.norm(Minima[i]-GroupedMinima[j][k])<0.1:
                    GroupedMinima[j].append(Minima[i])
                    GroupedTimeMinima[j].append(TimeMinimaOrbit_s[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedMinima)-1:
            GroupedMinima.append([])
            GroupedTimeMinima.append([])
            GroupedMinima[j+1].append(Minima[i])
            GroupedTimeMinima[j+1].append(TimeMinimaOrbit_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

Minima=[]
TimeMinimaOrbit_s=[]
for i in range(len(GroupedMinima)):

    Minima.append(np.mean(GroupedMinima[i], axis=0))
    TimeMinimaOrbit_s.append(np.min(GroupedTimeMinima[i]))


# Run CG and use the elemnts of Minima as staring points
for i in range(len(Minima)):
    #print( Minima[i], TimeMinimaOrbit_s[i]) 

    res = minimize(fct_Rastrigin, Minima[i], method='CG',options={'disp': False, 'maxiter':1000000})

    
    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            LocalMinimaO.append(sol)
            DistanceO.append(np.linalg.norm(Minima[i]-sol))
            TimeLocalMinimaO.append(TimeMinimaOrbit_s[i])

# group all elements from LocalMinimumO in group such that all ellemnts in one group are closer than 0.1 

GroupedLocalMinimaO=[]
GroupedTimeLocalMinimaO=[]
GroupedDistanceO=[]

for i in range(len(LocalMinimaO)):
    if i==0:
        GroupedLocalMinimaO.append([])
        GroupedTimeLocalMinimaO.append([])
        GroupedDistanceO.append([])
        
        GroupedLocalMinimaO[0].append(LocalMinimaO[i])
        GroupedTimeLocalMinimaO[0].append(TimeLocalMinimaO[i])
        GroupedDistanceO[0].append(DistanceO[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedLocalMinimaO)):
            for k in range(len(GroupedLocalMinimaO[j])):
                if np.linalg.norm(LocalMinimaO[i]-GroupedLocalMinimaO[j][k])<0.1:
                    GroupedLocalMinimaO[j].append(LocalMinimaO[i])
                    GroupedTimeLocalMinimaO[j].append(TimeLocalMinimaO[i])
                    GroupedDistanceO[j].append(DistanceO[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedLocalMinimaO)-1:
            GroupedLocalMinimaO.append([])
            GroupedTimeLocalMinimaO.append([])
            GroupedDistanceO.append([])

            GroupedLocalMinimaO[j+1].append(LocalMinimaO[i])
            GroupedTimeLocalMinimaO[j+1].append(TimeMinimaOrbit_s[i])
            GroupedDistanceO[j+1].append(DistanceO[i])
    

# create a new list of LocalMinimaO and TimeLocalMinimaO where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time
LocalMinimaO=[]
TimeLocalMinimaO=[]
DistanceO=[]
for i in range(len(GroupedLocalMinimaO)):
    LocalMinimaO.append(np.mean(GroupedLocalMinimaO[i], axis=0))
    TimeLocalMinimaO.append(np.min(GroupedTimeLocalMinimaO[i]))
    DistanceO.append(np.mean(GroupedDistanceO[i]))



# now we can calculate the element of NumberOfFoundLocalMinima from TimeLocalMinimaO 

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaO)):
        if TimeLocalMinimaO[i]<t:
            NumberOfFoundLocalMinima[t]+=1
            


np.save(path+'NumberOfFoundLocalMinima'+str(nameOrbit)+str(simulation)+'', NumberOfFoundLocalMinima)
np.save(path+'Distance'+str(nameOrbit)+str(simulation)+'', DistanceO)
np.save(path+'LocalMinima'+str(nameOrbit)+str(simulation)+'', LocalMinimaO)
np.save(path+'TimeLocalMinima'+str(nameOrbit)+str(simulation)+'', TimeLocalMinimaO)

#%% same for harmonic parameters

AverageNumberOfFoundLocalMinimaD=np.zeros(T_PSO)

TimeLocalMinimaD=[]
LocalMinimaD=[]
DistanceD=[]
NumberOfFoundLocalMinima=np.zeros(T_PSO)

TimeMinimaDamping_s=np.load(path+'TimesMinima_s' + str(nameHarmonic) + str(simulation) + '.npy')  
Minima=np.load(path+'Minima_s' + str(nameHarmonic) + str(simulation) + '.npy')

# create groups of elements from Minima such that all elements in one group are closer than 0.1

GroupedMinima=[]
GroupedTimeMinima=[]

for i in range(len(Minima)):
    if i==0:
        GroupedMinima.append([])
        GroupedTimeMinima.append([])
        
        GroupedMinima[0].append(Minima[i])
        GroupedTimeMinima[0].append(TimeMinimaDamping_s[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedMinima)):
            for k in range(len(GroupedMinima[j])):
                if np.linalg.norm(Minima[i]-GroupedMinima[j][k])<0.1:
                    GroupedMinima[j].append(Minima[i])
                    GroupedTimeMinima[j].append(TimeMinimaDamping_s[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedMinima)-1:
            GroupedMinima.append([])
            GroupedTimeMinima.append([])
            GroupedMinima[j+1].append(Minima[i])
            GroupedTimeMinima[j+1].append(TimeMinimaDamping_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

Minima=[]
TimeMinimaDamping_s=[]
for i in range(len(GroupedMinima)):

    Minima.append(np.mean(GroupedMinima[i], axis=0))
    TimeMinimaDamping_s.append(np.min(GroupedTimeMinima[i]))


# Run CG and use the elemnts of Minima as staring points
for i in range(len(Minima)):
        

    res = minimize(fct_Rastrigin, Minima[i], method='CG',options={'disp': False, 'maxiter':1000000})

    
    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            LocalMinimaD.append(sol)
            DistanceD.append(np.linalg.norm(Minima[i]-sol))
            TimeLocalMinimaD.append(TimeMinimaDamping_s[i])

# group all elements from LocalMinimumO in group such that all ellemnts in one group are closer than 0.1 

GroupedLocalMinimaD=[]
GroupedTimeLocalMinimaD=[]
GroupedDistanceD=[]

for i in range(len(LocalMinimaD)):
    if i==0:
        GroupedLocalMinimaD.append([])
        GroupedTimeLocalMinimaD.append([])
        GroupedDistanceD.append([])
        
        GroupedLocalMinimaD[0].append(LocalMinimaD[i])
        GroupedTimeLocalMinimaD[0].append(TimeLocalMinimaD[i])
        GroupedDistanceD[0].append(DistanceD[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedLocalMinimaD)):
            for k in range(len(GroupedLocalMinimaD[j])):
                if np.linalg.norm(LocalMinimaD[i]-GroupedLocalMinimaD[j][k])<0.1:
                    GroupedLocalMinimaD[j].append(LocalMinimaD[i])
                    GroupedTimeLocalMinimaD[j].append(TimeLocalMinimaD[i])
                    GroupedDistanceD[j].append(DistanceD[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedLocalMinimaD)-1:
            GroupedLocalMinimaD.append([])
            GroupedTimeLocalMinimaD.append([])
            GroupedDistanceD.append([])

            GroupedLocalMinimaD[j+1].append(LocalMinimaD[i])
            GroupedTimeLocalMinimaD[j+1].append(TimeMinimaDamping_s[i])
            GroupedDistanceD[j+1].append(DistanceD[i])
    

# create a new list of LocalMinimaO and TimeLocalMinimaO where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time
LocalMinimaD=[]
TimeLocalMinimaD=[]
DistanceD=[]
for i in range(len(GroupedLocalMinimaD)):
    LocalMinimaD.append(np.mean(GroupedLocalMinimaD[i], axis=0))
    TimeLocalMinimaD.append(np.min(GroupedTimeLocalMinimaD[i]))
    DistanceD.append(np.mean(GroupedDistanceD[i]))



# now we can calculate the element of NumberOfFoundLocalMinima from TimeLocalMinimaO 

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaD)):
        if TimeLocalMinimaD[i]<t:
            NumberOfFoundLocalMinima[t]+=1
            

    
    
       
np.save(path+'TimeLocalMinima'+str(nameHarmonic)+str(simulation)+'', TimeLocalMinimaD)
np.save(path+'Distance'+str(nameHarmonic)+str(simulation)+'', DistanceD)
np.save(path+'LocalMinima'+str(nameHarmonic)+str(simulation)+'', LocalMinimaD)
np.save(path+'NumberOfFoundLocalMinima'+str(nameHarmonic)+str(simulation)+'', NumberOfFoundLocalMinima)

#%%

# for classic

AverageNumberOfFoundLocalMinimaC=np.zeros(T_PSO)

TimeLocalMinimaC=[]
LocalMinimaC=[]
DistanceC=[]
NumberOfFoundLocalMinima=np.zeros(T_PSO)

TimeMinimaClassic_s=np.load(path+'TimesMinima_s' + str(nameClassic) + str(simulation) + '.npy')  
Minima=np.load(path+'Minima_s' + str(nameClassic) + str(simulation) + '.npy')

# create groups of elements from Minima such that all elements in one group are closer than 0.1

GroupedMinima=[]
GroupedTimeMinima=[]

for i in range(len(Minima)):
    if i==0:
        GroupedMinima.append([])
        GroupedTimeMinima.append([])
        
        GroupedMinima[0].append(Minima[i])
        GroupedTimeMinima[0].append(TimeMinimaClassic_s[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedMinima)):
            for k in range(len(GroupedMinima[j])):
                if np.linalg.norm(Minima[i]-GroupedMinima[j][k])<0.1:
                    GroupedMinima[j].append(Minima[i])
                    GroupedTimeMinima[j].append(TimeMinimaClassic_s[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedMinima)-1:
            GroupedMinima.append([])
            GroupedTimeMinima.append([])
            GroupedMinima[j+1].append(Minima[i])
            GroupedTimeMinima[j+1].append(TimeMinimaClassic_s[i])
    
# create a new list of Minima and TimeMinima where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time

Minima=[]
TimeMinimaClassic_s=[]
for i in range(len(GroupedMinima)):

    Minima.append(np.mean(GroupedMinima[i], axis=0))
    TimeMinimaClassic_s.append(np.min(GroupedTimeMinima[i]))


# Run CG and use the elemnts of Minima as staring points
for i in range(len(Minima)):
    

    res = minimize(fct_Rastrigin, Minima[i], method='CG',options={'disp': False, 'maxiter':1000000})

    
    sol=res.x
    if np.max(np.abs(gradient(sol)))<0.01:
        
        if hesse(sol)=='min':
            
            LocalMinimaC.append(sol)
            DistanceC.append(np.linalg.norm(Minima[i]-sol))
            TimeLocalMinimaC.append(TimeMinimaClassic_s[i])

# group all elements from LocalMinimumO in group such that all ellemnts in one group are closer than 0.1 

GroupedLocalMinimaC=[]
GroupedTimeLocalMinimaC=[]
GroupedDistanceC=[]

for i in range(len(LocalMinimaC)):
    if i==0:
        GroupedLocalMinimaC.append([])
        GroupedTimeLocalMinimaC.append([])
        GroupedDistanceC.append([])
        
        GroupedLocalMinimaC[0].append(LocalMinimaC[i])
        GroupedTimeLocalMinimaC[0].append(TimeLocalMinimaC[i])
        GroupedDistanceC[0].append(DistanceC[i])
    else:
        # find the right group for Minima[i]
        for j in range(len(GroupedLocalMinimaC)):
            for k in range(len(GroupedLocalMinimaC[j])):
                if np.linalg.norm(LocalMinimaC[i]-GroupedLocalMinimaC[j][k])<0.1:
                    GroupedLocalMinimaC[j].append(LocalMinimaC[i])
                    GroupedTimeLocalMinimaC[j].append(TimeLocalMinimaC[i])
                    GroupedDistanceC[j].append(DistanceC[i])
                    # break the j and k loop
                    break
            else:
                continue
            break
        # if no group create a new group
        if j==len(GroupedLocalMinimaC)-1:
            GroupedLocalMinimaC.append([])
            GroupedTimeLocalMinimaC.append([])
            GroupedDistanceC.append([])

            GroupedLocalMinimaC[j+1].append(LocalMinimaC[i])
            GroupedTimeLocalMinimaC[j+1].append(TimeMinimaClassic_s[i])
            GroupedDistanceC[j+1].append(DistanceC[i])
    

# create a new list of LocalMinimaO and TimeLocalMinimaO where all alements are unique, by calculalatign the mean elemnt of each group and the minimal time
LocalMinimaC=[]
TimeLocalMinimaC=[]
DistanceC=[]
for i in range(len(GroupedLocalMinimaC)):
    LocalMinimaC.append(np.mean(GroupedLocalMinimaC[i], axis=0))
    TimeLocalMinimaC.append(np.min(GroupedTimeLocalMinimaC[i]))
    DistanceC.append(np.mean(GroupedDistanceC[i]))



# now we can calculate the element of NumberOfFoundLocalMinima from TimeLocalMinimaO 

for t in range(T_PSO):
    for i in range(len(TimeLocalMinimaC)):
        if TimeLocalMinimaC[i]<t:
            NumberOfFoundLocalMinima[t]+=1

    
np.save(path+'Distance'+str(nameClassic)+str(simulation)+'', DistanceC)
np.save(path+'NumberOfFoundLocalMinima'+str(nameClassic)+str(simulation)+'', NumberOfFoundLocalMinima)
np.save(path+'TimeLocalMinima'+str(nameClassic)+str(simulation)+'', TimeLocalMinimaC)
np.save(path+'LocalMinima'+str(nameClassic)+str(simulation)+'', LocalMinimaC)



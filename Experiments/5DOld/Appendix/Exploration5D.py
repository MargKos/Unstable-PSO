import numpy as np
import matplotlib.pyplot as plt
from Variables import *

#%% average the number of found local minima that were calculated in Exploration5DMultiprocessing
T_PSO_s=200

def fct_average(Parameters, eps): # Paramerter: Damped, Overdamped, Divergent eps=radius of ball
    Average=np.zeros(T_PSO_s) # Average gives the average number of local minima found at time t over all simulations
    
    
    if Parameters=='A':
        name=nameA
        
    if Parameters=='B':
        name=nameB    
    
    if Parameters=='C':
        name=nameC

    # load the data of each simulation and average it
    for s in range(sim):
        Average=Average+np.load(path+'AllScores'+str(name)+str(eps)+str(s)+'.npy')*120 # multiplied by 120, if one wants the toal number instead of percentage
    
    
    return Average/sim


#%% Averages the number of calculated local minima for each radius eps, the array Eps has to correspond to the one in Exploration5DMultiprocessing5D.py

#Eps=np.arange(0,1+0.1,0.1)

#Eps=np.arange(0.05,0.4,0.05)
Eps=np.arange(0.05,0.4,0.025)
                        
for i in range(len(Eps)-1):
    
    AverageScoreA=fct_average('A', Eps[i])
    AverageScoreB=fct_average('B', Eps[i])
    AverageScoreC=fct_average('C', Eps[i])
    
    
    # save the results

    np.save('./Results/AverageScore'+str(nameA)+str(Eps[i]), AverageScoreA)
    
    np.save('./Results/AverageScore'+str(nameB)+str(Eps[i]), AverageScoreB)
    
    np.save('./Results/AverageScore'+str(nameC)+str(Eps[i]), AverageScoreC)
    
    
    
# Load local minima of Michalewicz function
 
LocalMinimas=np.load('./LocalMinima/LocalMinimaMichalewicz.npy', allow_pickle=True)

#%% Calculate the commulative number of local minima found over all simulations
    
def fct_local_minima(Parameters, sim, T_PSO, eps): # Parameters: Damped, Overdamped, Divergent, eps=radius of ball, sim=number of simualtions, T_PSO=number of iterations, eps=radius of ball

    AllScores = np.zeros(T_PSO) # AllScores gives nuber of local minima found till time t
    score = np.zeros(len(LocalMinimas)) # score is a binaer vector that is 1 if local minima was found and 0 else
    
    # select the right parameters
    
    if Parameters=='A':
        name=nameA
        
    if Parameters=='B':
        name=nameB    
        
    if Parameters=='C':
        name=nameC
    
    # Load the data of each simulation and store it efficiently

    Xsim=np.zeros((5,n, sim, T_PSO))
    for s in range(sim):
        X=np.load(path+'X_s'+str(name)+str(s)+'.npy')
        Xsim[:,:, s, :] = X[:,:, 0:T_PSO]

    for t in range(T_PSO):    
        for k in range(len(LocalMinimas)):
            if score[k] == 0:
                for s in range(sim):
                    # load particle positions of simulation s
                    X=np.load(path+'X_s'+str(name)+str(s)+'.npy')
                    # go thorugh all particles and check if they are in the ball of radius eps around the local minima k
                    for i in range(n):
                        dist = np.linalg.norm(Xsim[:,i, s, t] - LocalMinimas[k])
                        if dist < eps:
                            score[k] = 1
                            break
                    # to save time, if the local minima was found in one simulation, we can go to the next local minima and donÄt check other simulations
                    if score[k] > 0:
                        break
        # calculate the number of local minima found at time t from score
        NumberOfLocalMinima = np.count_nonzero(score) / len(LocalMinimas)
        AllScores[t] = NumberOfLocalMinima
    print('done'+str(name)+str(sim))
    np.save('./Results/SimScores'+str(name)+str(sim)+'.npy', AllScores)
        
    return AllScores

#%%
'''
SimulationsNumber=[20,50, 100, 200] # number of simulations for commulative number of local minima
for s in range(4):

    
    fct_local_minima('A', SimulationsNumber[s], 2000, 0.52)
    fct_local_minima('B', SimulationsNumber[s], 2000, 0.52)
    fct_local_minima('C', SimulationsNumber[s], 2000, 0.52)

'''

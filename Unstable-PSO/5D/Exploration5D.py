import numpy as np
import matplotlib.pyplot as plt
from Variables import *

#%% average the number of found local minima that were calculated in Exploration5DMultiprocessing

def fct_average(Parameters, eps): # Paramerter: Damped, Overdamped, Divergent eps=radius of ball
    Average=np.zeros(T_PSO) # Average gives the average number of local minima found at time t over all simulations

    if Parameters=='Damped':
        name=nameDamped
    
    if Parameters=='Overdamped':
        name=nameOverdamped
    
    if Parameters=='Divergent':
        name=nameDivergent

    # load the data of each simulation and average it
    for s in range(sim):
        Average=Average+np.load(path+'AllScores'+str(name)+str(eps)+str(s)+'.npy')*120 # multiplied by 120, if one wants the toal number instead of percentage
    
    
    return Average/sim


#%% Averages the number of calculated local minima for each radius eps, the array Eps has to correspond to the one in Exploration5DMultiprocessing5D.py

Eps=np.arange(0,1+0.1,0.1)
                        
for i in range(len(Eps)-1):

    AverageScoreDivergent=fct_average('Divergent', Eps[i+1])
    AverageScoreDamped=fct_average('Damped', Eps[i+1])
    AverageScoreOverdamped=fct_average('Overdamped', Eps[i+1]) 

    # save the results

    np.save('./Results/AverageScore'+str(nameDivergent)+str(Eps[i+1]),AverageScoreDivergent )
    
    np.save('./Results/AverageScore'+str(nameDamped)+str(Eps[i+1]), AverageScoreDamped)

    np.save('./Results/AverageScore'+str(nameOverdamped)+str(Eps[i+1]), AverageScoreOverdamped)
    
# Load local minima of Michalewicz function
 
LocalMinimas=np.load('./LocalMinima/LocalMinimaMichalewicz.npy', allow_pickle=True)

#%% Calculate the commulative number of local minima found over all simulations
    
def fct_local_minima(Parameters, sim, T_PSO, eps): # Parameters: Damped, Overdamped, Divergent, eps=radius of ball, sim=number of simualtions, T_PSO=number of iterations, eps=radius of ball

    AllScores = np.zeros(T_PSO) # AllScores gives nuber of local minima found till time t
    score = np.zeros(len(LocalMinimas)) # score is a binaer vector that is 1 if local minima was found and 0 else
    
    # select the right parameters

    if Parameters=='Damped':
        name=nameDamped
    
    if Parameters=='Overdamped':
        name=nameOverdamped
    
    if Parameters=='Divergent':
        name=Divergent
    
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
print('yes')
SimulationsNumber=[20,50, 100, 200] # number of simulations for commulative number of local minima
for s in range(4):

    fct_local_minima('Damped', SimulationsNumber[s], 2000, 0.52)
    fct_local_minima('Overdamped', SimulationsNumber[s], 2000, 0.52)
    fct_local_minima('Divergent', SimulationsNumber[s], 2000, 0.52)



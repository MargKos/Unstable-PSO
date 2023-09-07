import numpy as np
import matplotlib.pyplot as plt
from Variables import *


path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/Simulations/'

#%% load local minimas

LocalMinimaRastrigin=np.load('./LocalMinima/LocalMinimaRastrigin.npy') # fct number 1
LocalMinimaHM=np.load('./LocalMinima/LocalMinimaHM.npy') # fct number 5
LocalMinimaSchwefel=np.load('./LocalMinima/LocalMinimaSchwefel.npy') # fct number 6
LocalMinimaGriewank=np.load('./LocalMinima/LocalMinimaGriewank.npy') # fct number 7



###
#%% Count how many local minima were found till time t
def fct_local_minima(Parameters, LocalMinima,T_PSO): # Parameters: 'Harmonic', 'Classic', 'Orbit'; LocalMinima: LocalMinimaRastrigin, LocalMinimaSchwefel, LocalMinimaGriewank, LocalMinimaHM, T_PSO: number of iterations
    
    AverageScore=np.zeros(T_PSO) # average number of local minima found at time t
    AllScores=np.zeros((sim,T_PSO)) # number of local minima found at time t for each simulation
    
    # find the function number

    if str(LocalMinima)==str(LocalMinimaRastrigin):
        function_number=1
    
    if str(LocalMinima)==str(LocalMinimaHM):
        function_number=5
        
    if str(LocalMinima)==str(LocalMinimaSchwefel):
        function_number=6
    
    if str(LocalMinima)==str(LocalMinimaGriewank):
        function_number=7
        
    # print the function number
    print(function_number)

    # find the name w.r.t the parameters

    if Parameters=='Damped':
        name=nameDamped[function_number]
    
    if Parameters=='Overdamped':
        name=nameOverdamped[function_number]
    
    if Parameters=='Divergent':
        name=nameDivergent[function_number]
    
    # calculate the average number of local minima found at time t for each simulation

    for s in range(sim):

        # load all particles positions of simulation s

        X=np.load(path+'X_s'+str(name)+str(s)+'.npy')

        # score is a binaer vector with 1 if a local minima was found and zero if not

        score=np.zeros(len(LocalMinima))

        for t in range(T_PSO):
            for k in range(len(LocalMinima)):
                # check for lall ocal minima that were not found so far if some particle i foudn it at time t
                if score[k]==0:
                    # check for each particle
                    for  i in range(n):
                        if np.abs(X[0,i,t]-LocalMinima[k])<0.1:
                            score[k]=1
                            break
              
            NumberOfLocalMinima=np.count_nonzero(score)/len(LocalMinima)

            AllScores[s,t]=NumberOfLocalMinima

    # average over all simulations   

    for t in range(T_PSO):
        AverageScore[t]=np.mean(AllScores[:,t])
        
    np.save('./Results/Exploration'+str(name)+'.npy', AverageScore)
    return AverageScore
        


#%% Calculate exploration for all parameters and all multimodal functions

ExplorationDivergent=fct_local_minima('Divergent', LocalMinimaSchwefel,T_PSO)
ExplorationDamped=fct_local_minima('Damped', LocalMinimaSchwefel,T_PSO)
ExplorationOverdamped=fct_local_minima('Overdamped', LocalMinimaSchwefel,T_PSO)

ExplorationDivergent=fct_local_minima('Divergent', LocalMinimaRastrigin,T_PSO)
ExplorationDamped=fct_local_minima('Damped', LocalMinimaRastrigin,T_PSO)
ExplorationOverdamped=fct_local_minima('Overdamped', LocalMinimaRastrigin,T_PSO)

ExplorationDivergent=fct_local_minima('Divergent', LocalMinimaGriewank,T_PSO)
ExplorationDamped=fct_local_minima('Damped', LocalMinimaGriewank,T_PSO)
ExplorationOverdamped=fct_local_minima('Overdamped', LocalMinimaGriewank,T_PSO)

ExplorationDivergent=fct_local_minima('Divergent', LocalMinimaHM,T_PSO)
ExplorationDamped=fct_local_minima('Damped', LocalMinimaHM,T_PSO)
ExplorationOverdamped=fct_local_minima('Overdamped', LocalMinimaHM,T_PSO)
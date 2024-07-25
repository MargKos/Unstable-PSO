
import numpy as np
import sys
from Variables import *

# Print task number
if len(sys.argv) < 2:
    print('No input provided...')
    sys.exit()
else:
    simulation = int(sys.argv[1]) - 1
    print('\nHello! This is task simulation', simulation)

# Load local minima of Michalewicz function
 
LocalMinimas=np.load('./LocalMinima/LocalMinimaMichalewicz.npy', allow_pickle=True)

#%% Count how many local minima were found till time t
def fct_local_minima(Parameters,T_PSO, eps,s): # Parameters: Damped, Overdamped, Divergent, eps=radius of ball,s=number of simualtion

    # load particle positions

    if Parameters=='Damped':
        X=np.load(path+'X_s'+str(nameDamped)+str(simulation)+'.npy')
        name=nameDamped

    if Parameters=='Overdamped':
        X=np.load(path+'X_s'+str(nameOverdamped)+str(simulation)+'.npy')
        name=nameOverdamped
    
    if Parameters=='Divergent':
        X=np.load(path+'X_s'+str(nameDivergent)+str(simulation)+'.npy')
        name=nameDivergent
    
    if Parameters=='A':
        X=np.load(path+'X_s'+str(nameA)+str(simulation)+'.npy')
        name=nameA

    if Parameters=='B':
        X=np.load(path+'X_s'+str(nameB)+str(simulation)+'.npy')
        name=nameB
    
    if Parameters=='C':
        X=np.load(path+'X_s'+str(nameC)+str(simulation)+'.npy')
        name=nameC

    AllScores=np.zeros(T_PSO) # AllScores gives nuber of local minima found until time t
    score=np.zeros(len(LocalMinimas)) #score is a binaer vector that is 1 if local minima was found and 0 else

    for t in range(T_PSO):
        for k in range(len(LocalMinimas)):
            if score[k]==0:
                # check for each particle
                for i in range(n):
                    if np.linalg.norm(X[:,i,t]-LocalMinimas[k])<eps:
                        score[k]=1
                        break
          
        NumberOfLocalMinima=np.count_nonzero(score)/len(LocalMinimas)

        AllScores[t]=NumberOfLocalMinima
      
    np.save(path+'AllScores'+str(name)+str(eps)+str(s)+'.npy', AllScores)
    print('done exploration'+str(eps))
    
    

#%% Calculate the average number for different eps

# generate an array with different eps
#Eps=np.arange(0.05,0.4,0.05)
Eps=np.arange(0.05,0.4,0.025)
for i in range(len(Eps)): # loop of different radi
    fct_local_minima('Divergent',200,Eps[i],simulation)
    fct_local_minima('Overdamped',200,Eps[i],simulation) 
    fct_local_minima('Damped',200,Eps[i], simulation)
    fct_local_minima('A',200,Eps[i],simulation)
    fct_local_minima('B',200,Eps[i],simulation) 
    fct_local_minima('C',200,Eps[i], simulation)
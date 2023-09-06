
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
def fct_local_minima(Parameters,T_PSO, eps,s): # Parameters: Harmonic, Classic, Orbit, eps=radius of ball,s=number of simualtion

    # load particle positions

    if Parameters=='Harmonic':
        X=np.load(path+'X_s'+str(nameHarmonic)+str(simulation)+'.npy')
        name=nameHarmonic

    if Parameters=='Classic':
        X=np.load(path+'X_s'+str(nameClassic)+str(simulation)+'.npy')
        name=nameClassic
    
    if Parameters=='Orbit':
        X=np.load(path+'X_s'+str(nameOrbit)+str(simulation)+'.npy')
        name=nameOrbit

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

Eps=np.arange(0,1+0.1,0.1)

for i in range(len(Eps)-1): # loop of different radi
    fct_local_minima('Orbit',T_PSO,Eps[i+1],simulation)
    fct_local_minima('Classic',T_PSO,Eps[i+1],simulation) 
    fct_local_minima('Harmonic',T_PSO,Eps[i+1], simulation)
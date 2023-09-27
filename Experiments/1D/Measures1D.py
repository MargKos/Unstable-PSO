import numpy as np
from Functions import *
import matplotlib.pyplot as plt
from Variables import *

# calculates the mean function value of the swarm at time t for each function

for function_number in range(8):

    fct=FunctionList[function_number]
    
    #%% Mean Divergent 
            
    LossDivergent=np.zeros(T_PSO) # LossO gives the mean function value of the swarm at time t
    for t in range(T_PSO):
        
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            DivergentG=np.load(path+'G_s'+str(nameDivergent[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(DivergentG[:,t])
            List.append(functionvalue)
        
        # calculate the mean function value of the swarm at time t for each function
        LossDivergent[t]=np.mean(List)

    np.save('./Results/Mean'+str(nameDivergent[function_number])+'', LossDivergent)


    #%% Mean Damped
            
    LossDamped=np.zeros(T_PSO)
    for t in range(T_PSO):
        
        fct=FunctionList[function_number]
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            DampedG=np.load(path+'G_s'+str(nameDamped[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(DampedG[:,t])
            List.append(functionvalue)

        # calculate the mean function value of the swarm at time t for each function   
        LossDamped[t]=np.mean(List)

    # save the results
    np.save('./Results/Mean'+str(nameDamped[function_number])+'', LossDamped)

    #%% Mean Overdamped
            
    LossOverdamped=np.zeros(T_PSO)
    for t in range(T_PSO):
        
        fct=FunctionList[function_number]
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            OverdampedG=np.load(path+'G_s'+str(nameOverdamped[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(OverdampedG[:,t])
            List.append(functionvalue)
        # calculate the mean function value of the swarm at time t for each function
        LossOverdamped[t]=np.mean(List)

    np.save('./Results/Mean'+str(nameOverdamped[function_number])+'', LossOverdamped)

    #%% Concentration of particles around globale best

    def fct_concentration(X, Gl,  eps): # X=particle positions, Gl=global best position, eps=radius of ball
        Concentration=np.zeros(T_PSO)
        for t in range(T_PSO):
            closeparticles=0 # counts the number of particles that are close to the global best
            for i in range(n):
                if np.linalg.norm(X[:,i,t]-Gl[:,t])<eps: # check if particle i is in the ball around the global best
                    closeparticles=closeparticles+1 # if yes, increase the counter
            # calculate the concentration of particles around the global best at time t
            Concentration[t]=closeparticles/n
        
        return Concentration

    #%%
    
    ConcentrationDamped=np.zeros(T_PSO)
    ConcentrationOverdamped=np.zeros(T_PSO)
    ConcentrationDivergent=np.zeros(T_PSO)
    for s in range(sim):
        # load global best position of simulation s and positions of all particles

        G_Damped=np.load(path+'G_s'+str(nameDamped[function_number])+str(s)+'.npy')
        G_Overdamped=np.load(path+'G_s'+str(nameOverdamped[function_number])+str(s)+'.npy')
        G_Divergent=np.load(path+'G_s'+str(nameDivergent[function_number])+str(s)+'.npy')

        X_Damped=np.load(path+'X_s'+str(nameDamped[function_number])+str(s)+'.npy')
        X_Overdamped=np.load(path+'X_s'+str(nameOverdamped[function_number])+str(s)+'.npy')
        X_Divergent=np.load(path+'X_s'+str(nameDivergent[function_number])+str(s)+'.npy')

        # calculate the concentration of particles around the global best
    
        ConcentrationDamped=ConcentrationDamped+fct_concentration(X_Damped, G_Damped, 0.1)
        ConcentrationOverdamped=ConcentrationOverdamped+fct_concentration(X_Overdamped, G_Overdamped, 0.1)
        ConcentrationDivergent=ConcentrationDivergent+fct_concentration(X_Divergent, G_Divergent, 0.1)
    #%%
    np.save('./Results/AveragedConcentration'+str(nameDivergent[function_number])+'.npy', ConcentrationDivergent/sim)
    np.save('./Results/AveragedConcentration'+str(nameOverdamped[function_number])+'.npy', ConcentrationOverdamped/sim)
    np.save('./Results/AveragedConcentration'+str(nameDamped[function_number])+'.npy', ConcentrationDamped/sim)


    
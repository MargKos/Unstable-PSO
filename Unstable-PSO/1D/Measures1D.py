import numpy as np
from Functions import *
import matplotlib.pyplot as plt
from Variables import *

# calculates the mean function value of the swarm at time t for each function

for function_number in range(8):

    fct=FunctionList[function_number]
    
    #%% Mean Orbit 
            
    LossO=np.zeros(T_PSO) # LossO gives the mean function value of the swarm at time t
    for t in range(T_PSO):
        
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            OrbitG=np.load(path+'G_s'+str(nameOrbit[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(OrbitG[:,t])
            List.append(functionvalue)
        
        # calculate the mean function value of the swarm at time t for each function
        LossO[t]=np.mean(List)

    np.save('./Results/Mean'+str(nameOrbit[function_number])+'', LossO)


    #%% Mean Harmonic
            
    LossH=np.zeros(T_PSO)
    for t in range(T_PSO):
        
        fct=FunctionList[function_number]
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            HarmonicG=np.load(path+'G_s'+str(nameHarmonic[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(HarmonicG[:,t])
            List.append(functionvalue)

        # calculate the mean function value of the swarm at time t for each function   
        LossH[t]=np.mean(List)

    # save the results
    np.save('./Results/Mean'+str(nameHarmonic[function_number])+'', LossH)

    #%% Mean Classic
            
    LossC=np.zeros(T_PSO)
    for t in range(T_PSO):
        
        fct=FunctionList[function_number]
        List=[]
        for s in range(sim):
            # load global best position of simulation s
            ClassicG=np.load(path+'G_s'+str(nameClassic[function_number])+str(s)+'.npy')
            # evaluate function at global best position
            functionvalue=fct(ClassicG[:,t])
            List.append(functionvalue)
        # calculate the mean function value of the swarm at time t for each function
        LossC[t]=np.mean(List)

    np.save('./Results/Mean'+str(nameClassic[function_number])+'', LossC)

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
    
    ConcentrationHarmonic=np.zeros(T_PSO)
    ConcentrationClassic=np.zeros(T_PSO)
    ConcentrationOrbit=np.zeros(T_PSO)
    for s in range(sim):
        # load global best position of simulation s and positions of all particles

        G_Harmonic=np.load(path+'G_s'+str(nameHarmonic[function_number])+str(s)+'.npy')
        G_Classic=np.load(path+'G_s'+str(nameClassic[function_number])+str(s)+'.npy')
        G_Orbit=np.load(path+'G_s'+str(nameOrbit[function_number])+str(s)+'.npy')

        X_Harmonic=np.load(path+'X_s'+str(nameHarmonic[function_number])+str(s)+'.npy')
        X_Classic=np.load(path+'X_s'+str(nameClassic[function_number])+str(s)+'.npy')
        X_Orbit=np.load(path+'X_s'+str(nameOrbit[function_number])+str(s)+'.npy')

        # calculate the concentration of particles around the global best
    
        ConcentrationHarmonic=ConcentrationHarmonic+fct_concentration(X_Harmonic, G_Harmonic, 0.1)
        ConcentrationClassic=ConcentrationClassic+fct_concentration(X_Classic, G_Classic, 0.1)
        ConcentrationOrbit=ConcentrationOrbit+fct_concentration(X_Orbit, G_Orbit, 0.1)
    #%%
    np.save('./Results/AveragedConcentration'+str(nameOrbit[function_number])+'.npy', ConcentrationOrbit/sim)
    np.save('./Results/AveragedConcentration'+str(nameClassic[function_number])+'.npy', ConcentrationClassic/sim)
    np.save('./Results/AveragedConcentration'+str(nameHarmonic[function_number])+'.npy', ConcentrationHarmonic/sim)


    
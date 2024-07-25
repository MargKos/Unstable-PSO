import numpy as np
from Functions import *

# Learning Parameters



def C():
    
    w=0.85
    mu=1.4
    
    return w, mu

def A():

    mu=2.2
    w=0.7
    
    return w, mu

def B():
    
    w=0.95
    mu=1
    
    return w, mu


w_C, mu_C=C()
w_B, mu_B=B()
w_A, mu_A=A()



# Simulation Parameters

n=20 # number of particles
T_PSO=20000 # number of timesteps

sim=200 # has to fit to the batch file
function_number=8 # number of the Michalewicz function
dim=5
fct_loss=FunctionList[function_number]

StartWerte=np.load('Uniform5D.npy') # uniform starting points


# Define Names

nameC=str(function_number)+str('C')+'Mu'+str(mu_C)+'Particle'+str(n)+'T'+str(T_PSO)
nameA=str(function_number)+str('A')+'Mu'+str(mu_A)+'Particle'+str(n)+'T'+str(T_PSO)
nameB=str(function_number)+str('B')+'Mu'+str(mu_B)+'Particle'+str(n)+'T'+str(T_PSO)

# path to save the simulations

path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/Data5D/'



def Divergent():
    
    w=1
    mu=0.03
    
    return w, mu

def Damped():

    mu=0.3215
    w=0.958058
    
    return w, mu

def Overdamped():
    
    w=0.7
    mu=1.4
    
    return w, mu



w_divergent, mu_divergent=Divergent()
w_damped, mu_damped=Damped()
w_overdamped, mu_overdamped=Overdamped()



# Simulation Parameters

n=20 # number of particles
T_PSO=20000 # number of timesteps

sim=200 # has to fit to the batch file
function_number=8 # number of the Michalewicz function
dim=5
fct_loss=FunctionList[function_number]

StartWerte=np.load('Uniform5D.npy') # uniform starting points


# Define Names



nameDivergent=str(function_number)+str('Divergent')+'Mu'+str(mu_divergent)+'Particle'+str(n)+'T'+str(T_PSO)
nameOverdamped=str(function_number)+str('Overdamped')+'Mu'+str(mu_overdamped)+'Particle'+str(n)+'T'+str(T_PSO)
nameDamped=str(function_number)+str('Damped')+'Mu'+str(mu_damped)+'Particle'+str(n)+'T'+str(T_PSO)

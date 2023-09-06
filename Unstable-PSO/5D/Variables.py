import numpy as np
from Functions import *

# Learning Parameters


def Orbit():
    
    w=1
    mu=0.03
    
    return w, mu

def Harmonic():

    mu=0.3215
    w=0.958058
    
    return w, mu

def Classic():
    
    w=0.7
    mu=1.4
    
    return w, mu


w_o, mu_o=Orbit()
w_h, mu_h=Harmonic()
w_c, mu_c=Classic()


# Simulation Parameters

n=20 # number of particles
T_PSO=2000 # number of timesteps

sim=100 # has to fit to the batch file
function_number=8 # number of the Michalewicz function
dim=5
fct_loss=FunctionList[function_number]

StartWerte=np.load('Uniform5D.npy') # uniform starting points


# Define Names


nameOrbit=str(function_number)+str('Orbit')+'Mu'+str(mu_o)+'Particle'+str(n)+'T'+str(T_PSO)
nameClassic=str(function_number)+str('Classic')+'Mu'+str(mu_h)+'Particle'+str(n)+'T'+str(T_PSO)
nameHarmonic=str(function_number)+str('Harmonic')+'Mu'+str(mu_c)+'Particle'+str(n)+'T'+str(T_PSO)

# path to save the simulations

path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/SimulationsTest2/'
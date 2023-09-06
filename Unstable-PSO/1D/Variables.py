import numpy as np

# simulation settings


n=10 # number of particles
T_PSO=2000 # number of timesteps
sim=100    # number of simulations
dim=1


# rates

def Orbit():
    
    w=1
    mu=0.14035087719298245
   
    
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

# generate names for each test function, to save results

nameOrbit=[]
nameClassic=[]
nameHarmonic=[]

for function_number in range(8):
    nameOrbit.append(str(function_number)+str('Orbit')+'Mu'+str(mu_o)+'Particle'+str(n)+'T'+str(T_PSO))
    nameClassic.append(str(function_number)+str('Classic')+'Mu'+str(mu_c)+'Particle'+str(n)+'T'+str(T_PSO))
    nameHarmonic.append(str(function_number)+str('Harmonic')+'Mu'+str(mu_h)+'Particle'+str(n)+'T'+str(T_PSO))


# define path where to save simulations
path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/SimulationsTest/'

# load starting points for each test function
StartingPoints=np.load('BoundaryOld1D.npy')


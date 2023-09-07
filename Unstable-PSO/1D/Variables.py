import numpy as np

# simulation settings


n=10 # number of particles
T_PSO=2000 # number of timesteps
sim=100    # number of simulations
dim=1


# rates

def Divergent():
    
    w=1
    mu=0.14035087719298245
   
    
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

# generate names for each test function, to save results

nameDivergent=[]
nameOverdamped=[]
nameDamped=[]

for function_number in range(8):
    nameDivergent.append(str(function_number)+str('Divergent')+'Mu'+str(mu_divergent)+'Particle'+str(n)+'T'+str(T_PSO))
    nameOverdamped.append(str(function_number)+str('Overdamped')+'Mu'+str(mu_overdamped)+'Particle'+str(n)+'T'+str(T_PSO))
    nameDamped.append(str(function_number)+str('Damped')+'Mu'+str(mu_damped)+'Particle'+str(n)+'T'+str(T_PSO))


# define path where to save simulations
path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/SimulationsTest/'

# load starting points for each test function
StartingPoints=np.load('BoundaryOld1D.npy')


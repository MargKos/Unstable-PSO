import numpy as np


def Divergent():
    w=1
    mu=0.002

    return w, mu,

def Damped():
    mu=0.3215
    w=0.958058

    return w, mu

def Overdamped():
    w=0.7
    mu=1.4

    return w, mu



w_divergent, mu_divergent=Divergent()
w_overdamped, mu_overdamped=Overdamped()
w_damped, mu_damped=Damped()

#%%

n=20 # number of particlses
T_PSO=500000 # number of iterations for PSO only saving global best postions and steady local best positions
T_PSO_short=10000 # number of iterations in PSO which saves all positions
dim=30 # dimension
sim=100 # number of simulations, has to fit to the number assigned in the -sh file

# names to store the data

nameDivergent=str('Divergent')+'Mu'+str(mu_divergent)+'Particle'+str(n)+'T'+str(T_PSO)
nameOverdamped=str('Overdamped')+'Mu'+str(mu_overdamped)+'Particle'+str(n)+'T'+str(T_PSO)
nameDamped=str('Damped')+'Mu'+str(mu_damped)+'Particle'+str(n)+'T'+str(T_PSO)

# path to save
path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/Data30D/'
#%%

StartWerte=np.load('Boundary30D.npy')

#%%
# Waiting times to locate the local minima, has to be determined before 
counter_divergent=7
counter_overdamped=7
counter_damped=7



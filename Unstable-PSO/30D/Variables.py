import numpy as np


def Orbit():
    w=1
    mu=0.002

    return w, mu,

def Harmonic():
    mu=0.3215
    w=0.958058

    return w, mu

def Classic():
    w=0.7
    mu=1.4

    return w, mu



w_o, mu_o=Orbit()
w_c, mu_c=Classic()
w_h, mu_h=Harmonic()

#%%

n=20 # number of particlses
T_PSO=50000 # number of iterations for PSO only saving global best postions and steady local best positions
T_PSO_short=1000 # number of iterations in PSO which saves all positions
dim=30
sim=100


nameOrbit=str('Orbit')+'Mu'+str(mu_o)+'Particle'+str(n)+'T'+str(T_PSO)
nameClassic=str('Classic')+'Mu'+str(mu_c)+'Particle'+str(n)+'T'+str(T_PSO)
nameHarmonic=str('Harmonic')+'Mu'+str(mu_h)+'Particle'+str(n)+'T'+str(T_PSO)


# define variables
path='/home/htc/bzfkostr/SCRATCH/SimulationsPSO/SimulationsTest3/'
#%%

StartWerte=np.load('Boundary30D.npy')

#%%
# Waiting time
counter_Orbit=7
counter_Classic=7
counter_Harmonic=7



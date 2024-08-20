# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:41:50 2024

@author: mkost
"""

import numpy as np
import matplotlib.pyplot as plt
from Variables import *
from Functions import *

MeanA=np.load('./Results/Mean'+str(nameA)+'.npy')
MeanB=np.load('./Results/Mean'+str(nameB)+'.npy')
MeanC=np.load('./Results/Mean'+str(nameC)+'.npy')

#%%
ConcentrationA=np.load('./Results/AverageConcentration'+str(nameA)+'.npy')
ConcentrationB=np.load('./Results/AverageConcentration'+str(nameB)+'.npy')
ConcentrationC=np.load('./Results/AverageConcentration'+str(nameC)+'.npy')


MeanDivergent=np.load('./Results/Mean'+str(nameDivergent)+'.npy')
MeanOverdamped=np.load('./Results/Mean'+str(nameOverdamped)+'.npy')
MeanDamped=np.load('./Results/Mean'+str(nameDamped)+'.npy')

ConcentrationDivergent=np.load('./Results/AverageConcentration'+str(nameDivergent)+'.npy')
ConcentrationOverdamped=np.load('./Results/AverageConcentration'+str(nameOverdamped)+'.npy')
ConcentrationDamped=np.load('./Results/AverageConcentration'+str(nameDamped)+'.npy')

#%%
# Calculate the average exploration and the commulative exploration of last time step 
#Eps=np.arange(0,1+0.1,0.1) # radi of average exploration
#Eps=np.arange(0.05,0.4,0.05)
Eps=np.arange(0.05,0.4,0.025)
AverageExplorationA=np.zeros(14) # average exploration at last timestep for Divergent
AverageExplorationB=np.zeros(14) # average exploration at last timestep for Overdamped
AverageExplorationC=np.zeros(14) # average exploration at last timestep for damped

AverageExplorationDivergent=np.zeros(14) # average exploration at last timestep for Divergent
AverageExplorationOverdamped=np.zeros(14) # average exploration at last timestep for Overdamped
AverageExplorationDamped=np.zeros(14) # average exploration at last timestep for damped

Names=['C', 'A','B']
Names=['A', 'B','C']
for i in range(10):
       
    ExplorationA=np.load('./Results/AverageScore'+str(nameA)+str(Eps[i])+'.npy')
    ExplorationB=np.load('./Results/AverageScore'+str(nameB)+str(Eps[i])+'.npy')
    ExplorationC=np.load('./Results/AverageScore'+str(nameC)+str(Eps[i])+'.npy')

    AverageExplorationA[i]=ExplorationA[-1]
    AverageExplorationB[i]=ExplorationB[-1]
    AverageExplorationC[i]=ExplorationC[-1]
    
    
    ExplorationDivergent=np.load('./Results/AverageScore'+str(nameDivergent)+str(Eps[i])+'.npy')
    ExplorationOverdamped=np.load('./Results/AverageScore'+str(nameOverdamped)+str(Eps[i])+'.npy')
    ExplorationDamped=np.load('./Results/AverageScore'+str(nameDamped)+str(Eps[i])+'.npy')

    AverageExplorationDivergent[i]=ExplorationDivergent[-1]
    AverageExplorationOverdamped[i]=ExplorationOverdamped[-1]
    AverageExplorationDamped[i]=ExplorationDamped[-1]


#%%

eps=0.3
SimulationsNumber=[20,50, 100, 200] # should fit to the numbers in Exploration5D.py
T_PSO_s=2000

SimDivergent=np.zeros(4) # commulative exploration of Divergent at last timestep
SimOverdamped=np.zeros(4) # commulative exploration of Overdamped at last timestep
SimDamped=np.zeros(4) # commulative exploration of Damped at last timestep

SimA=np.zeros(4) # commulative exploration of Divergent at last timestep
SimB=np.zeros(4) # commulative exploration of Overdamped at last timestep
SimC=np.zeros(4) # commulative exploration of Damped at last timestep

for s in range(4):
    
    # Load the commulative exploration 
    ExplorationSimA=np.load('./Results/SimScores'+str(nameA)+str(SimulationsNumber[s])+'.npy')
    ExplorationSimB=np.load('./Results/SimScores'+str(nameB)+str(SimulationsNumber[s])+'.npy')
    ExplorationSimC=np.load('./Results/SimScores'+str(nameC)+str(SimulationsNumber[s])+'.npy')
    
    # Calculate the commulative exploration at last timestep
    SimA[s]=ExplorationSimA[-1]
    SimB[s]=ExplorationSimB[-1]
    SimC[s]=ExplorationSimC[-1]
    
    # Load the commulative exploration 
    ExplorationDivergentSim=np.load('./Results/SimScores'+str(nameDivergent)+str(SimulationsNumber[s])+'.npy')
    ExplorationOverdampedSim=np.load('./Results/SimScores'+str(nameOverdamped)+str(SimulationsNumber[s])+'.npy')
    ExplorationDampedSim=np.load('./Results/SimScores'+str(nameDamped)+str(SimulationsNumber[s])+'.npy')
    
    # Calculate the commulative exploration at last timestep
    SimDivergent[s]=ExplorationDivergentSim[-1]
    SimOverdamped[s]=ExplorationOverdampedSim[-1]
    SimDamped[s]=ExplorationDampedSim[-1]

#%%
Random=np.load('./Results/Random.npy')



#%% Paper Plot


legend_fontsize=30
T_PSO_s=170 # time for plotting of Mean
T_PSO_s2=1000 # time for plotting of Concentration
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.7, bottom=0.22,left=0.1, right=0.95)

axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOverdamped[0:T_PSO_s], linewidth=2.0,color='orange')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDivergent[0:T_PSO_s],  linewidth=2.0,color='blue')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDamped[0:T_PSO_s], linewidth=2.0,color='green')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOverdamped[0:T_PSO_s],'s',markersize=9,markevery=20, color='orange',label='Overdamped')
axs[0].plot( np.linspace(1, T_PSO_s, T_PSO_s), MeanDamped[0:T_PSO_s],'^', markersize=7,markevery=20,color='green',label='Damped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDivergent[0:T_PSO_s],'o' ,markersize=7, markevery=20,color='blue',  label='Divergent O')
 
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanA[0:T_PSO_s],'v',markersize=9,markevery=20, color='red',label='Divergent'+' '+str(Names[0]))
axs[0].plot( np.linspace(1, T_PSO_s, T_PSO_s), MeanB[0:T_PSO_s],'1', markersize=7,markevery=20,color='magenta',label='Divergent'+' '+str(Names[1]))
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanC[0:T_PSO_s],'p' ,markersize=7, markevery=20,color='grey',  label='Divergent '+' '+str(Names[2]))
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanA[0:T_PSO_s], linewidth=2.0,color='red')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanB[0:T_PSO_s],  linewidth=2.0,color='magenta')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanC[0:T_PSO_s], linewidth=2.0,color='grey')


axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('mean value',fontsize=30)
axs[0].set_title('Mean Function Value',fontsize=30)

axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationOverdamped[0:T_PSO_s2], linewidth=2,color='orange')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationDamped[0:T_PSO_s2],linewidth=2.0,color='green')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationDivergent[0:T_PSO_s2], linewidth=2.0,color='blue')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationOverdamped[0:T_PSO_s2],'s',markersize=9,markevery=46,color='orange')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationDamped[0:T_PSO_s2],'^',markersize=9,markevery=46,color='green')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationDivergent[0:T_PSO_s2],'o',markersize=9,markevery=46,color='blue')

axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationA[0:T_PSO_s2],'v',markersize=9,markevery=46,color='red')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationB[0:T_PSO_s2],'1',markersize=9,markevery=46,color='magenta')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationC[0:T_PSO_s2],'p',markersize=9,markevery=46,color='grey')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationA[0:T_PSO_s2],color='red')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationB[0:T_PSO_s2],color='magenta')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationC[0:T_PSO_s2],color='grey')

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_title('Exploitation',fontsize=30)
axs[1].set_ylabel('concentration', fontsize=30)
axs[0].legend(bbox_to_anchor=(0.3, 1.32, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize}, markerscale=3)
axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right',  fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
plt.savefig('./Plots/Fig9.png')


#%% Paper Plot




T_PSO_s = 4000  # time for plotting of Mean
start = 100
T_PSO_s2 = 1000  # time for plotting of Concentration

fig, axs = plt.subplots(1, 1, figsize=(20, 15))
plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.88)

axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanOverdamped[start:T_PSO_s], linewidth=2.0, color='orange')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanDivergent[start:T_PSO_s], linewidth=2.0, color='blue')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanDamped[start:T_PSO_s], linewidth=2.0, color='green')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanOverdamped[start:T_PSO_s], 's', markersize=9, markevery=200,
         color='orange', label='Overdamped')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanDamped[start:T_PSO_s], '^', markersize=7, markevery=200,
         color='green', label='Damped')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanDivergent[start:T_PSO_s], 'o', markersize=7, markevery=200,
         color='blue', label='Divergent O')

axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanA[start:T_PSO_s], 'v', markersize=9, markevery=200,
         color='red', label='Divergent ' + ' ' + str(Names[0]))
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanB[start:T_PSO_s], '1', markersize=7, markevery=200,
         color='magenta', label='Divergent ' + ' ' + str(Names[1]))
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanC[start:T_PSO_s], 'p', markersize=7, markevery=200,
         color='grey', label='Divergent ' + ' ' + str(Names[2]))
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanA[start:T_PSO_s], linewidth=2.0, color='red')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanB[start:T_PSO_s], linewidth=2.0, color='magenta')
axs.plot(np.linspace(start, T_PSO_s, T_PSO_s - start), MeanC[start:T_PSO_s], linewidth=2.0, color='grey')

axs.tick_params(axis='x', labelsize=30)
axs.tick_params(axis='y', labelsize=30)
axs.set_xlabel('time', fontsize=30)
axs.set_ylabel('mean value', fontsize=30)
axs.set_title('Mean Function Value', fontsize=30)

axs.legend(bbox_to_anchor=(-0.08, 1.1, 1.1, 0.1), loc="center",
               ncol=3,  prop={'size': 25}, markerscale=3)

plt.savefig('./Plots/FigB2.png')

#%%


import matplotlib.pyplot as plt
TI=9
#fig, axs = plt.subplots(1, 2, figsize=(15,5))  # Adjusted figure size
#plt.subplots_adjust(top=0.7, bottom=0.22, left=0.1, right=0.95)

fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.7, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(Eps[0:TI], AverageExplorationOverdamped[0:TI], 's', markersize=9, color='orange', label='Overdamped')
axs[0].plot(Eps[0:TI], AverageExplorationDamped[0:TI], '^', markersize=7, color='green', label='Damped')
axs[0].plot(Eps[0:TI], AverageExplorationDivergent[0:TI], 'o', markersize=7, color='blue', label='Divergent O')

axs[0].plot(Eps[0:TI], AverageExplorationA[0:TI], 'v', markersize=9, color='red', label='Divergent' + ' ' + str(Names[0]))
axs[0].plot(Eps[0:TI], AverageExplorationB[0:TI], '1', markersize=7, color='magenta', label='Divergent' + ' ' + str(Names[1]))
axs[0].plot(Eps[0:TI], AverageExplorationC[0:TI], 'p', markersize=7, color='grey', label='Divergent' + ' ' + str(Names[2]))

axs[0].plot(Eps[0:TI], AverageExplorationOverdamped[0:TI], color='orange')
axs[0].plot(Eps[0:TI], AverageExplorationDivergent[0:TI], color='blue')
axs[0].plot(Eps[0:TI], AverageExplorationDamped[0:TI], color='green')

axs[0].plot(Eps[0:TI], AverageExplorationA[0:TI], color='red')
axs[0].plot(Eps[0:TI], AverageExplorationB[0:TI], color='magenta')
axs[0].plot(Eps[0:TI], AverageExplorationC[0:TI], color='grey')
#axs[0].text(1, 70, 'Random Search', fontsize=25, va='center', ha='left')
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].plot(Eps[0:TI], Random[0:TI], color='black')
axs[0].plot(Eps[0:TI], Random[0:TI], 'o', markevery=1, color='black')
axs[0].set_xlabel(r'$\epsilon_\theta$', fontsize=30)
axs[0].set_ylabel('number of found minima', fontsize=30)

axs[1].plot(SimulationsNumber, SimOverdamped * 120, 's', markersize=9, color='orange')
axs[1].plot(SimulationsNumber, SimDamped * 120, '^', markersize=7, color='green')
axs[1].plot(SimulationsNumber, SimDivergent * 120, 'o', markersize=7, color='blue')
axs[1].plot(SimulationsNumber, SimOverdamped * 120, color='orange')
axs[1].plot(SimulationsNumber, SimDivergent * 120, color='blue')
axs[1].plot(SimulationsNumber, SimDamped * 120, color='green')

axs[1].plot(SimulationsNumber, SimA * 120, 'v', markersize=9, color='red')
axs[1].plot(SimulationsNumber, SimB * 120, '1', markersize=7, color='magenta')
axs[1].plot(SimulationsNumber, SimC * 120, 'p', markersize=7, color='grey')
axs[1].plot(SimulationsNumber, SimA * 120, color='red')
axs[1].plot(SimulationsNumber, SimB * 120, color='magenta')
axs[1].plot(SimulationsNumber, SimC * 120, color='grey')


axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('number of simulations', fontsize=30)
#axs[1].set_ylabel('number of found minima', fontsize=25)
axs[0].legend(bbox_to_anchor=(0.3, 1.32, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize}, markerscale=3)
plt.xticks([20, 50, 100, 200])

# Adding text next to the black line in the second subplot
axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')

plt.savefig('./Plots/Fig9.png')  # Adjusted dpi value

import numpy as np
import matplotlib.pyplot as plt
from Variables import *
from Functions import *

# Code for plotting the results of the 5D Michalewicz function: mean, exploitation, average exploration and commulative exploration

#%% Load the mean function values and the concentration
MeanDivergent=np.load('./Results/Mean'+str(nameDivergent)+'.npy')
MeanOverdamped=np.load('./Results/Mean'+str(nameOverdamped)+'.npy')
MeanDamped=np.load('./Results/Mean'+str(nameDamped)+'.npy')

ConcentrationDivergent=np.load('./Results/AverageConcentration'+str(nameDivergent)+'.npy')
ConcentrationOverdamped=np.load('./Results/AverageConcentration'+str(nameOverdamped)+'.npy')
ConcentrationDamped=np.load('./Results/AverageConcentration'+str(nameDamped)+'.npy')


#%% Plot mean and concentration in one figure
legend_fontsize=30
T_PSO_s=170 # time for plotting of Mean
T_PSO_s2=400 # time for plotting of Concentration
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOverdamped[0:T_PSO_s], linewidth=2.0,color='orange')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDivergent[0:T_PSO_s],  linewidth=2.0,color='blue')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDamped[0:T_PSO_s], linewidth=2.0,color='green')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOverdamped[0:T_PSO_s],'s',markersize=9,markevery=20, color='orange',label='Overdamped')
axs[0].plot( np.linspace(1, T_PSO_s, T_PSO_s), MeanDamped[0:T_PSO_s],'^', markersize=7,markevery=20,color='green',label='Damped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanDivergent[0:T_PSO_s],'o' ,markersize=7, markevery=20,color='blue',  label='Divergent')

axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('mean value',fontsize=30)
axs[0].set_title('Mean Function Value',fontsize=30)

axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationOverdamped[0:T_PSO_s2], linewidth=2,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationDamped[0:T_PSO_s2],linewidth=2.0,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationDivergent[0:T_PSO_s2], linewidth=2.0,color='blue',  label='Divergent')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationOverdamped[0:T_PSO_s2],'s',markersize=9,markevery=46,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationDamped[0:T_PSO_s2],'s',markersize=9,markevery=46,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationDivergent[0:T_PSO_s2],'s',markersize=9,markevery=46,color='blue',  label='Divergent')

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_title('Exploitation',fontsize=30)
axs[1].set_ylabel('concentration', fontsize=30)
axs[0].legend(bbox_to_anchor=(0.2, 1.1, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})
axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right',  fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
plt.savefig('./Plots/Fig8.png')


#%%
# Calculate the average exploration and the commulative exploration of last time step 
Eps=np.arange(0,1+0.1,0.1) # radi of average exploration
AverageExplorationDivergent=np.zeros(10) # average exploration at last timestep for Divergent
AverageExplorationOverdamped=np.zeros(10) # average exploration at last timestep for Overdamped
AverageExplorationDamped=np.zeros(10) # average exploration at last timestep for damped
for i in range(10):
       
    ExplorationDivergent=np.load('./Results/AverageScore'+str(nameDivergent)+str(Eps[i+1])+'.npy')
    ExplorationOverdamped=np.load('./Results/AverageScore'+str(nameOverdamped)+str(Eps[i+1])+'.npy')
    ExplorationDamped=np.load('./Results/AverageScore'+str(nameDamped)+str(Eps[i+1])+'.npy')

    AverageExplorationDivergent[i]=ExplorationDivergent[-1]
    AverageExplorationOverdamped[i]=ExplorationOverdamped[-1]
    AverageExplorationDamped[i]=ExplorationDamped[-1]



eps=0.3
SimulationsNumber=[20,50, 100, 200] # should fit to the numbers in Exploration5D.py
T_PSO_s=2000
SimDivergent=np.zeros(4) # commulative exploration of Divergent at last timestep
SimOverdamped=np.zeros(4) # commulative exploration of Overdamped at last timestep
SimDamped=np.zeros(4) # commulative exploration of Damped at last timestep
for s in range(4):
    
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

#%% Plot the average exploration and the commulative exploration of last time step in ine Figure
legend_fontsize=30
SimulationsNumber=[20,50, 100, 200]
T_PSO_s=170

fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)

axs[0].plot(Eps[1:11],AverageExplorationOverdamped, 's',markersize=9, color='orange', label='Overdamped')
axs[0].plot(Eps[1:11],AverageExplorationDamped,'^', markersize=7, color='green', label='Damped')
axs[0].plot(Eps[1:11],AverageExplorationDivergent, 'o', markersize=7,color='blue', label='Divergent')
axs[0].plot(Eps[1:11],AverageExplorationOverdamped,  color='orange')
axs[0].plot(Eps[1:11],AverageExplorationDivergent, color='blue')
axs[0].plot(Eps[1:11],AverageExplorationDamped, color='green')
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel(r'$\epsilon_\theta$',  fontsize=30)
axs[0].set_ylabel('number of found minima',fontsize=25)


axs[1].plot(SimulationsNumber,SimOverdamped*120, 's',markersize=9, color='orange', label='Overdamped')
axs[1].plot(SimulationsNumber,SimDamped*120,'^', markersize=7, color='green', label='Damped')
axs[1].plot(SimulationsNumber,SimDivergent*120, 'o', markersize=7,color='blue', label='Divergent')
axs[1].plot(SimulationsNumber,SimOverdamped*120,  color='orange')
axs[1].plot(SimulationsNumber,SimDivergent*120, color='blue')
axs[1].plot(SimulationsNumber,SimDamped*120, color='green')
axs[1].plot(np.linspace(1,200,200),Random*120, color='black')
axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('number of simulations', fontsize=25)
axs[1].set_ylabel('number of found minima', fontsize=25)
axs[0].legend(bbox_to_anchor=(0.2, 1.02, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})
plt.xticks([20, 50, 100, 200])
# Adding text next to the black line in the second subplot
axs[1].text(100, 70, 'Random Search', fontsize=25, va='center', ha='left')
axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right',  fontweight='bold')
plt.savefig('./Plots/Fig9.png')


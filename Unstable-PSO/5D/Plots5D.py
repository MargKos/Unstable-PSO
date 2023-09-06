import numpy as np
import matplotlib.pyplot as plt
from Variables import *
from Functions import *

# Code for plotting the results of the 5D Michalewicz function: mean, exploitation, average exploration and commulative exploration

#%% Load the mean function values and the concentration
MeanOrbit=np.load('./Results/Mean'+str(nameOrbit)+'.npy')
MeanClassic=np.load('./Results/Mean'+str(nameClassic)+'.npy')
MeanHarmonic=np.load('./Results/Mean'+str(nameHarmonic)+'.npy')

ConcentrationOrbit=np.load('./Results/AverageConcentration'+str(nameOrbit)+'.npy')
ConcentrationClassic=np.load('./Results/AverageConcentration'+str(nameClassic)+'.npy')
ConcentrationHarmonic=np.load('./Results/AverageConcentration'+str(nameHarmonic)+'.npy')


#%% Plot mean and concentration in one figure
legend_fontsize=30
T_PSO_s=170 # time for plotting of Mean
T_PSO_s2=400 # time for plotting of Concentration
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanClassic[0:T_PSO_s], linewidth=2.0,color='orange')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOrbit[0:T_PSO_s],  linewidth=2.0,color='blue')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanHarmonic[0:T_PSO_s], linewidth=2.0,color='green')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanClassic[0:T_PSO_s],'s',markersize=9,markevery=20, color='orange',label='Classic')
axs[0].plot( np.linspace(1, T_PSO_s, T_PSO_s), MeanHarmonic[0:T_PSO_s],'^', markersize=7,markevery=20,color='green',label='Damped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), MeanOrbit[0:T_PSO_s],'o' ,markersize=7, markevery=20,color='blue',  label='Divergent Oscillator')

axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('mean value',fontsize=30)
axs[0].set_title('Mean Function Value',fontsize=30)

axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationClassic[0:T_PSO_s2], linewidth=2,color='orange',label='Classic')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationHarmonic[0:T_PSO_s2],linewidth=2.0,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationOrbit[0:T_PSO_s2], linewidth=2.0,color='blue',  label='Divergent Oscillator')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationClassic[0:T_PSO_s2],'s',markersize=9,markevery=46,color='orange',label='Classic')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2), ConcentrationHarmonic[0:T_PSO_s2],'s',markersize=9,markevery=46,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_s2, T_PSO_s2),ConcentrationOrbit[0:T_PSO_s2],'s',markersize=9,markevery=46,color='blue',  label='Divergent Oscillator')

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_title('Exploitation',fontsize=30)
axs[1].set_ylabel('concentration', fontsize=30)
axs[0].legend(bbox_to_anchor=(0.2, 1.1, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})
axs[0].text(0.54, -0.25, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right')
axs[1].text(0.54, -0.25, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right')
plt.savefig('./Plots/Fig5.eps')

#%% Create a figure that consists of 2x2 subplots and plot the exploration for different radi
'''
Radi=[1,3,5,10]
l=0
cols=2
rows=2
Eps=np.arange(0,1+0.1,0.1)
legend_fontsize=20
fig, axs = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows))
plt.subplots_adjust(top=0.89, bottom=0.15,left=0.1, right=0.95)
for i in range(rows):
    for j in range(cols):
        axsi=axs[i,j]
        ExplorationOrbit=np.load('./Results/AverageScore'+str(nameOrbit)+str(Eps[Radi[l]])+'.npy')
        ExplorationClassic=np.load('./Results/AverageScore'+str(nameClassic)+str(Eps[Radi[l]])+'.npy')
        ExplorationHarmonic=np.load('./Results/AverageScore'+str(nameHarmonic)+str(Eps[Radi[l]])+'.npy')
        vol=8/15*(Eps[Radi[l]]**5)/(np.pi**3)*100
        axsi.plot(np.linspace(1,T_PSO,T_PSO), ExplorationClassic[0:T_PSO], color='orange', label='Classic')
        axsi.plot(np.linspace(1,T_PSO,T_PSO), ExplorationOrbit[0:T_PSO], color='blue', label='Divergent Oscillator')
        axsi.plot(np.linspace(1,T_PSO,T_PSO), ExplorationHarmonic[0:T_PSO], color='green', label='Damped')
        axsi.set_title('v = '+str(np.round(vol, decimals=5))+str('%'), fontsize=20)
        axsi.set_ylim(0,120)
        axsi.tick_params(axis='x', labelsize=20)
        axsi.tick_params(axis='y', labelsize=20)
        l=l+1
        
axs[0,1].legend(bbox_to_anchor=(-0.8, 1.02, 1, 0.3), loc="upper left",
               ncol=4,  prop={'size': legend_fontsize})

# Adjust y-axis label position
fig.text(0.02, 0.58, 'number of found local minima', fontsize=20, va='center', rotation='vertical', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Adjust x-axis label position
fig.text(0.52, 0.05, 'time', fontsize=20, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


plt.savefig('./Plots/'+str(nameOrbit)+'Exploration.eps')


#%% Create a figure that consists of 2x2 subplots and plor the exploration for different number of simulations
l=0
cols=2
rows=2
eps=0.3
SimulationsNumber=[20,50, 100, 200]
Eps=np.arange(0,1+0.1,0.1)
T_PSO_s=2000
legend_fontsize=20
fig, axs = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows))
plt.subplots_adjust(top=0.89, bottom=0.15,left=0.1, right=0.95)
for i in range(rows):
    for j in range(cols):
        axsi=axs[i,j]
        ExplorationOrbit=np.load('./Results/SimScores'+str(nameOrbit)+str(SimulationsNumber[l])+'.npy')
        ExplorationHarmonic=np.load('./Results/SimScores'+str(nameHarmonic)+str(SimulationsNumber[l])+'.npy')
        ExplorationClassic=np.load('./Results/SimScores'+str(nameClassic)+str(SimulationsNumber[l])+'.npy')
        
        vol=8/15*(Eps[3]**5)/(np.pi**3)*100
        axsi.plot(np.linspace(1,T_PSO_s,T_PSO_s), ExplorationClassic[0:T_PSO_s]*120, color='orange', label='Classic')
        axsi.plot(np.linspace(1,T_PSO_s,T_PSO_s), ExplorationOrbit[0:T_PSO_s]*120, color='blue', label='Divergent Oscillator')
        axsi.plot(np.linspace(1,T_PSO_s,T_PSO_s), ExplorationHarmonic[0:T_PSO_s]*120, color='green', label='Damped')
       
        axsi.set_title('s = '+str(SimulationsNumber[l]), fontsize=20)
        axsi.set_ylim(0,120)
        axsi.tick_params(axis='x', labelsize=20)
        axsi.tick_params(axis='y', labelsize=20)
       
        l=l+1
       
 
axs[0,1].legend(bbox_to_anchor=(-0.8, 1.02, 1, 0.3), loc="upper left",
               ncol=4,  prop={'size': legend_fontsize})

# Adjust y-axis label position
fig.text(0.02, 0.58, 'number of found local minima', fontsize=20, va='center', rotation='vertical', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Adjust x-axis label position
fig.text(0.52, 0.05, 'time', fontsize=20, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))



plt.savefig('./Plots/Sim'+str(nameOrbit)+'Exploration.eps')
'''
#%%
# Calculate the average exploration and the commulative exploration of last time step 
Eps=np.arange(0,1+0.1,0.1) # radi of average exploration
AverageExplorationO=np.zeros(10) # average exploration at last timestep for Orbit
AverageExplorationC=np.zeros(10) # average exploration at last timestep for Classic
AverageExplorationH=np.zeros(10) # average exploration at last timestep for Harmonic
for i in range(10):
       
    ExplorationOrbit=np.load('./Results/AverageScore'+str(nameOrbit)+str(Eps[i+1])+'.npy')
    ExplorationClassic=np.load('./Results/AverageScore'+str(nameClassic)+str(Eps[i+1])+'.npy')
    ExplorationHarmonic=np.load('./Results/AverageScore'+str(nameHarmonic)+str(Eps[i+1])+'.npy')

    AverageExplorationO[i]=ExplorationOrbit[-1]
    AverageExplorationC[i]=ExplorationClassic[-1]
    AverageExplorationH[i]=ExplorationHarmonic[-1]

eps=0.3
SimulationsNumber=[20,50, 100, 200] # should fit to the numbers in Exploration5D.py
T_PSO_s=2000
SimO=np.zeros(4) # commulative exploration of Orbit at last timestep
SimC=np.zeros(4) # commulative exploration of Classic at last timestep
SimH=np.zeros(4) # commulative exploration of Harmonic at last timestep
for s in range(4):
    
    # Load the commulative exploration 
    ExplorationOrbitSim=np.load('./Results/SimScores'+str(nameOrbit)+str(SimulationsNumber[s])+'.npy')
    ExplorationClassicSim=np.load('./Results/SimScores'+str(nameClassic)+str(SimulationsNumber[s])+'.npy')
    ExplorationHarmonicSim=np.load('./Results/SimScores'+str(nameHarmonic)+str(SimulationsNumber[s])+'.npy')
    
    # Calculate the commulative exploration at last timestep
    SimO[s]=ExplorationOrbitSim[-1]
    SimC[s]=ExplorationClassicSim[-1]
    SimH[s]=ExplorationHarmonicSim[-1]


#%% Plot the average exploration and the commulative exploration of last time step in ine Figure
legend_fontsize=30
SimulationsNumber=[20,50, 100, 200]
T_PSO_s=170
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.2,left=0.1, right=0.95)
axs[0].plot(Eps[1:11],AverageExplorationC, 's',markersize=9, color='orange', label='Classic')
axs[0].plot(Eps[1:11],AverageExplorationH,'^', markersize=7, color='green', label='Damped')
axs[0].plot(Eps[1:11],AverageExplorationO, 'o', markersize=7,color='blue', label='Divergent Oscillator')
axs[0].plot(Eps[1:11],AverageExplorationC,  color='orange')
axs[0].plot(Eps[1:11],AverageExplorationO, color='blue')
axs[0].plot(Eps[1:11],AverageExplorationH, color='green')
axs[0].tick_params(axis='x', labelsize=26)
axs[0].tick_params(axis='y', labelsize=26)
axs[0].set_xlabel(r'$\epsilon_\theta$',  fontsize=26)
axs[0].set_ylabel('number of found minima',fontsize=26)


axs[1].plot(SimulationsNumber,SimC*120, 's',markersize=9, color='orange', label='Classic')
axs[1].plot(SimulationsNumber,SimH*120,'^', markersize=7, color='green', label='Damped')
axs[1].plot(SimulationsNumber,SimO*120, 'o', markersize=7,color='blue', label='Divergent Oscillator')
axs[1].plot(SimulationsNumber,SimC*120,  color='orange')
axs[1].plot(SimulationsNumber,SimO*120, color='blue')
axs[1].plot(SimulationsNumber,SimH*120, color='green')
axs[1].tick_params(axis='x', labelsize=26)
axs[1].tick_params(axis='y', labelsize=26)
axs[1].set_xlabel('number of simulations', fontsize=26)
axs[1].set_ylabel('number of found minima', fontsize=26)
axs[0].legend(bbox_to_anchor=(0.2, 1.02, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})
plt.xticks([20, 50, 100, 200])
axs[0].text(0.54, -0.22, '(a)', transform=axs[0].transAxes, fontsize=26, va='top', ha='right')
axs[1].text(0.54, -0.22, '(b)', transform=axs[1].transAxes, fontsize=26, va='top', ha='right')
plt.savefig('./Plots/Fig6.eps')


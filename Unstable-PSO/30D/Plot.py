#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:54:52 2023

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from Variables import *


#%%
MeanOrbit=np.load('./Results/ShortMeanRastrigin'+str(nameOrbit)+''+'.npy')
MeanHarmonic=np.load('./Results/ShortMeanRastrigin'+str(nameHarmonic)+''+'.npy')
MeanClassic=np.load('./Results/ShortMeanRastrigin'+str(nameClassic)+''+'.npy')
legend_fontsize=30


#%% Plot Concentration



ConcentrationOrbit=np.load('./Results/AverageConcentrationRastrigin'+str(nameOrbit)+'.npy')
ConcentrationClassic=np.load('./Results/AverageConcentrationRastrigin'+str(nameClassic)+'.npy')
ConcentrationHarmonic=np.load('./Results/AverageConcentrationRastrigin'+str(nameHarmonic)+'.npy')


#%% Plot mean and concentration 
delta=1
#T_PSO_short=1000
T_PSO_short_c=1000
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanClassic[0:T_PSO_short],linewidth=2.0,color='orange')
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanOrbit[0:T_PSO_short], linewidth=2.0,color='blue')
axs[0].plot( np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanHarmonic[0:T_PSO_short],linewidth=2.0,color='green')

axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanClassic[0:T_PSO_short],'s',markersize=9,markevery=50,color='orange',label='Overdamped')
axs[0].plot( np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanHarmonic[0:T_PSO_short],'^',markersize=7,markevery=50,color='green',label='Damped')
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanOrbit[0:T_PSO_short],'o',markersize=7,markevery=50, color='blue',  label='Divergent')

axs[0].set_title('Mean Function Value',fontsize=30)
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('mean value',fontsize=30)
#axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#axs[0].xaxis.offsetText.set_fontsize(20)
axs[0].set_xlabel('number of iterations', fontsize=30)

axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationClassic[0:T_PSO_short_c],linewidth=2.0,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationHarmonic[0:T_PSO_short_c],linewidth=2.0,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c),ConcentrationOrbit[0:T_PSO_short_c],linewidth=2.0,color='blue',  label='Divergent')


axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationClassic[0:T_PSO_short_c],'s',markersize=9,markevery=50,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationHarmonic[0:T_PSO_short_c],'s',markersize=7,markevery=50,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c),ConcentrationOrbit[0:T_PSO_short_c],'s',markersize=7,markevery=50,color='blue',  label='Divergent')

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_ylabel('concentration', fontsize=30)
axs[1].set_title('Exploitation', fontsize=30)
axs[0].legend(bbox_to_anchor=(0.2, 1.1, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})

axs[0].text(0.54, -0.25, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right')
axs[1].text(0.54, -0.25, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right')
plt.savefig('./Plots/Fig7.eps')


#%% Ploter average number of found local minima


AverageNumberOfFoundLocalMinimaC=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameClassic)+'.npy')
AverageNumberOfFoundLocalMinimaH=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameHarmonic)+'.npy')
AverageNumberOfFoundLocalMinimaO=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameOrbit)+'.npy')


#%% Make a histogram of the function values of the local minima with bars that have no filling


AllLocalMinimaOValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameOrbit)+'.npy')
AllLocalMinimaHValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameHarmonic)+'.npy')
AllLocalMinimaCValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameClassic)+'.npy')


#%%
T_PSO_s=1000

fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaC[0:T_PSO_s], color='orange')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), AverageNumberOfFoundLocalMinimaO[0:T_PSO_s], color='blue')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaH[0:T_PSO_s], color='green')

axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), AverageNumberOfFoundLocalMinimaC[0:T_PSO_s],'s',markersize=7,markevery=100, color='orange', label='Overdamped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaH[0:T_PSO_s],'^',markersize=7,markevery=100, color='green', label='Damped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaO[0:T_PSO_s],'o', markersize=7,markevery=100, color='blue', label='Divergent')


axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('number of found minima',fontsize=30)
axs[0].legend(fontsize=20)

#create histogram of the vales of the local minima

axs[1].hist(AllLocalMinimaCValue,histtype='step', bins=50,color='orange',label='Overdamped')
axs[1].hist(AllLocalMinimaHValue,histtype='step', bins=50,color='green',label='Damped')
axs[1].hist(AllLocalMinimaOValue,histtype='step', bins=50,color='blue',  label='Divergent')

# 
a=plt.hist(AllLocalMinimaCValue,histtype='step', bins=50,color='orange',label='Overdamped')
F_C=int(a[0][0])

a=plt.hist(AllLocalMinimaHValue,histtype='step', bins=50,color='green',label='Damped')
F_H=int(a[0][0])

a=plt.hist(AllLocalMinimaOValue,histtype='step', bins=50,color='blue',  label='Divergent ')
F_O=int(a[0][0])

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('local minima value', fontsize=30)
axs[1].set_ylabel('frequency', fontsize=30)

axs[0].legend(bbox_to_anchor=(0.2, 1.02, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})
axs[0].text(0.54, -0.25, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right')
axs[1].text(0.54, -0.25, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right')

# mark the value of the lowest local minima in the histogram of damped parameters with a vertical line

axs[1].axvline(x=AllLocalMinimaHValue.min(), color='green', linestyle='--', linewidth=3)
axs[1].axvline(x=AllLocalMinimaCValue.min(), color='orange', linestyle='--', linewidth=3)
axs[1].axvline(x=AllLocalMinimaOValue.min(), color='blue', linestyle='--', linewidth=3)

#% write next to the line the correcpodning value from the histogram hist(AllLocalMinimaHValue,histtype='step', bins=50)

axs[1].text(AllLocalMinimaHValue.min()+0.5, 100, str(F_H), fontsize=26, va='top', ha='left', color='green')
axs[1].text(AllLocalMinimaCValue.min()+0.5, 100, str(F_C), fontsize=26, va='top', ha='left', color='orange')
axs[1].text(AllLocalMinimaOValue.min()+0.5, 100, str(F_O), fontsize=26, va='top', ha='left', color='blue')


plt.savefig('./Plots/Fig8.eps')


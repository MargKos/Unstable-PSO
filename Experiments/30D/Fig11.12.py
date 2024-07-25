#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:54:52 2023

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from Variables import *

''' Generate Figures'''

#%% load mean function values
MeanDivergent=np.load('./Results/ShortMeanRastrigin'+str(nameDivergent)+''+'.npy')
MeanDamped=np.load('./Results/ShortMeanRastrigin'+str(nameDamped)+''+'.npy')
MeanOverdamped=np.load('./Results/ShortMeanRastrigin'+str(nameOverdamped)+''+'.npy')
legend_fontsize=30


#%% load exploitation


ConcentrationDivergent=np.load('./Results/AverageConcentrationRastrigin'+str(nameDivergent)+'.npy')
ConcentrationOverdamped=np.load('./Results/AverageConcentrationRastrigin'+str(nameOverdamped)+'.npy')
ConcentrationDamped=np.load('./Results/AverageConcentrationRastrigin'+str(nameDamped)+'.npy')


#%% Plot mean and concentration in one Figure
delta=1
T_PSO_short=1000
T_PSO_short_c=1000
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanOverdamped[0:T_PSO_short],linewidth=2.0,color='orange')
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanDivergent[0:T_PSO_short], linewidth=2.0,color='blue')
axs[0].plot( np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanDamped[0:T_PSO_short],linewidth=2.0,color='green')
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanOverdamped[0:T_PSO_short],'s',markersize=9,markevery=50,color='orange',label='Overdamped')
axs[0].plot( np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanDamped[0:T_PSO_short],'^',markersize=7,markevery=50,color='green',label='Damped')
axs[0].plot(np.linspace(1, T_PSO_short, int(T_PSO_short/delta)), MeanDivergent[0:T_PSO_short],'o',markersize=7,markevery=50, color='blue',  label='Divergent Oscillator')
axs[0].set_title('Mean Function Value',fontsize=30)
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('mean value',fontsize=30)
axs[0].set_xlabel('number of iterations', fontsize=30)

axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationOverdamped[0:T_PSO_short_c],linewidth=2.0,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationDamped[0:T_PSO_short_c],linewidth=2.0,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c),ConcentrationDivergent[0:T_PSO_short_c],linewidth=2.0,color='blue',  label='Divergent Oscillator')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationOverdamped[0:T_PSO_short_c],'s',markersize=9,markevery=50,color='orange',label='Overdamped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c), ConcentrationDamped[0:T_PSO_short_c],'s',markersize=7,markevery=50,color='green',label='Damped')
axs[1].plot(np.linspace(1, T_PSO_short_c, T_PSO_short_c),ConcentrationDivergent[0:T_PSO_short_c],'s',markersize=7,markevery=50,color='blue',  label='Divergent Oscillator')
axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_ylabel('concentration', fontsize=30)
axs[1].set_title('Exploitation', fontsize=30)
axs[0].legend(bbox_to_anchor=(0.2, 1.1, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize},  markerscale=3)

axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right',  fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
plt.savefig('./Plots/Fig11.png')


#%% Plot average number of found local minima

# load the average number of found local minima

AverageNumberOfFoundLocalMinimaOverdamped=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameOverdamped)+'.npy')
AverageNumberOfFoundLocalMinimaDamped=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameDamped)+'.npy')
AverageNumberOfFoundLocalMinimaDivergent=np.load('./Results/AverageNumberOfFoundLocalMinima'+str(nameDivergent)+'.npy')

# load their values

AllLocalMinimaDivergentValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameDivergent)+'.npy')
AllLocalMinimaDampedValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameDamped)+'.npy')
AllLocalMinimaOverdampedValue=np.load('./Results/ValuesLocalMinimaRastrigin'+str(nameOverdamped)+'.npy')
AllLocalMinimaRandomValue=np.load('./Results/ValuesLocalMinimaRastriginRandom.npy')
#%% Plot average number of found local minima and their values in one plot
T_PSO_s=1000
T_PSO_s=400

fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaOverdamped[0:T_PSO_s], color='orange')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), AverageNumberOfFoundLocalMinimaDivergent[0:T_PSO_s], color='blue')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaDamped[0:T_PSO_s], color='green')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s), AverageNumberOfFoundLocalMinimaOverdamped[0:T_PSO_s],'s',markersize=7,markevery=100, color='orange', label='Overdamped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaDamped[0:T_PSO_s],'^',markersize=7,markevery=100, color='green', label='Damped')
axs[0].plot(np.linspace(1, T_PSO_s, T_PSO_s),AverageNumberOfFoundLocalMinimaDivergent[0:T_PSO_s],'o', markersize=7,markevery=100, color='blue', label='Divergent ')

#axs[0].plot(np.linspace(0, 1000,100), 19.83*np.ones(100), linewidth=5, color='black')

x_values = [7 * i for i in range(int(T_PSO_s/7)+1)]
heights=np.load('./Results/RandomSamplingshortsim57.npy')[0:int(T_PSO_s/7)+1]
axs[0].step(x_values, heights, where='post', linewidth=2, color='black')
axs[0].text(600, 25, 'Random Search', fontsize=25, va='center', ha='left')

axs[0].set_xlim(0,T_PSO_s)
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_xlabel('time',  fontsize=30)
axs[0].set_ylabel('number of found minima',fontsize=30)
axs[0].legend(fontsize=20)

#create histogram of the vales of the local minima

axs[1].hist(AllLocalMinimaOverdampedValue,histtype='step', bins=50,color='orange',label='Overdamped')
axs[1].hist(AllLocalMinimaDampedValue,histtype='step', bins=50,color='green',label='Damped')
axs[1].hist(AllLocalMinimaDivergentValue,histtype='step', bins=50,color='blue',  label='Divergent Oscillator')
axs[1].hist(AllLocalMinimaRandomValue,histtype='step', bins=50,color='black')
# 
a=plt.hist(AllLocalMinimaOverdampedValue,histtype='step', bins=50,color='orange',label='Overdamped')
F_C=int(a[0][0])

a=plt.hist(AllLocalMinimaDampedValue,histtype='step', bins=50,color='green',label='Damped')
F_H=int(a[0][0])

a=plt.hist(AllLocalMinimaDivergentValue,histtype='step', bins=50,color='blue',  label='Divergent')
F_O=int(a[0][0])

a=plt.hist(AllLocalMinimaRandomValue,histtype='step', bins=50,color='black',  label='Divergent')
F_R=int(a[0][0])

axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('local minima value', fontsize=30)
axs[1].set_ylabel('frequency', fontsize=30)

axs[1].set_ylim(0,180)
axs[0].legend(bbox_to_anchor=(0.2, 1.02, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize}, markerscale=3)

axs[0].text(100, 30, 'Random Approach',fontsize=20, fontweight='bold')
axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right',  fontweight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right', fontweight='bold')
# mark the value of the lowest local minima in the histogram of damped parameters with a vertical line

axs[1].axvline(x=AllLocalMinimaDampedValue.min(), color='green', linestyle='--', linewidth=3)
axs[1].axvline(x=AllLocalMinimaOverdampedValue.min(), color='orange', linestyle='--', linewidth=3)
axs[1].axvline(x=AllLocalMinimaDivergentValue.min(), color='blue', linestyle='--', linewidth=3)
axs[1].axvline(x=AllLocalMinimaRandomValue.min(), color='black', linestyle='-', linewidth=3)
#% write next to the line the correcpodning value from the histogram hist(AllLocalMinimaHValue,histtype='step', bins=50)

axs[1].text(AllLocalMinimaDampedValue.min()+0.5, 100, str(F_H), fontsize=26, va='top', ha='left', color='green')
axs[1].text(AllLocalMinimaOverdampedValue.min()+0.5, 100, str(F_C), fontsize=26, va='top', ha='left', color='orange')
axs[1].text(AllLocalMinimaDivergentValue.min()+2, 100, str(F_O), fontsize=26, va='top', ha='left', color='blue')
axs[1].text(AllLocalMinimaRandomValue.min()+0.5, 150, str(F_R), fontsize=26, va='top', ha='left', color='black')


plt.savefig('./Plots/Fig12.png')

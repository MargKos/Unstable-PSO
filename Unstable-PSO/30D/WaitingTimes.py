
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from scipy.optimize import curve_fit
from Variables import *
import matplotlib.ticker as ticker

'''Calculates the waiting times and their quantiles to locate the local minima for the three different parameters '''

#%% Collect the waiting times for the three different parameters in one list over all simulations and particles

OrbitWaitingTimes=[] # waiting times of the Orbit parameters
ClassicWaitingTimes=[] # waiting times of the Classic parameters
HarmonicWaitingTimes=[] # waiting times of the Harmonic parameters
for s in range(sim):   
    HarmonicWaitingTimes_s=np.load(path+'NumberofIterations_s' + str(nameHarmonic) + str(s) + '.npy', allow_pickle=True)
    for i in range(n):
        for t in range(len(HarmonicWaitingTimes_s[i])):
            HarmonicWaitingTimes.append(HarmonicWaitingTimes_s[i][t])

#%%
for s in range(sim):  
    ClassicWaitingTimes_s=np.load(path+'NumberofIterations_s' + str(nameClassic) + str(s) + '.npy', allow_pickle=True)
    for i in range(n):
        for t in range(len(ClassicWaitingTimes_s[i])):
            ClassicWaitingTimes.append(ClassicWaitingTimes_s[i][t])
#%%
for s in range(sim): 
    OrbitWaitingTimes_s=np.load(path+'NumberofIterations_s' + str(nameOrbit) + str(s) + '.npy', allow_pickle=True)
    for i in range(n):
        for t in range(len(OrbitWaitingTimes_s[i])):
            OrbitWaitingTimes.append(OrbitWaitingTimes_s[i][t])


#%% To  reduce the size of the data set, take a random sample of 50000 waiting times for each parameter configuration

OrbitWaitingTimesSample=np.random.choice(OrbitWaitingTimes, 50000)
ClassicWaitingTimesSample=np.random.choice(ClassicWaitingTimes, 50000)

#%% 
HarmonicWaitingTimesSample=[]

# alternative code

for i in range(50000):
    s=np.random.randint(0, len(HarmonicWaitingTimes)-1)
    HarmonicWaitingTimesSample.append(HarmonicWaitingTimes[s])

HarmonicWaitingTimesSample=np.array(HarmonicWaitingTimesSample)


#%%  

# save results

np.save('./Results/OrbitWaitingTimesSample.npy', OrbitWaitingTimesSample)
np.save('./Results/ClassicWaitingTimesSample.npy', ClassicWaitingTimesSample)
np.save('./Results/HarmonicWaitingTimesSample.npy', HarmonicWaitingTimesSample)

#%%
# load results

OrbitWaitingTimesSample=np.load('./Results/OrbitWaitingTimesSample.npy')
ClassicWaitingTimesSample=np.load('./Results/ClassicWaitingTimesSample.npy')
HarmonicWaitingTimesSample=np.load('./Results/HarmonicWaitingTimesSample.npy')


#%% Waiting times smaller than 1000

OrbitWaitingTimesSample=OrbitWaitingTimesSample[OrbitWaitingTimesSample<1000]
ClassicWaitingTimesSample=ClassicWaitingTimesSample[ClassicWaitingTimesSample<1000]
HarmonicWaitingTimesSample=HarmonicWaitingTimesSample[HarmonicWaitingTimesSample<1000]

#%% Waiting times larger than 0


ClassicWaitingTimesSample=ClassicWaitingTimesSample[ClassicWaitingTimesSample>0]
HarmonicWaitingTimesSample=HarmonicWaitingTimesSample[HarmonicWaitingTimesSample>0]
OrbitWaitingTimesSample=OrbitWaitingTimesSample[OrbitWaitingTimesSample>0]

#%% calculate the 0.95 quantile of ClassicWaitingTimesSample, OrbitWaitingTimesSample and HarmonicWaitingTimesSample

quantileC=np.quantile(ClassicWaitingTimesSample, 0.95)
quantileO=np.quantile(OrbitWaitingTimesSample, 0.95)
quantileH=np.quantile(HarmonicWaitingTimesSample, 0.95)

print('Classic', quantileC, 'Harmonic',quantileH,'Orbit', quantileO)

#%% Fit HarmonicWaitingTimes to Cauchy distribution 
# Fit a Cauchy distribution to the data:
paramH = cauchy.fit(HarmonicWaitingTimesSample)
paramC = cauchy.fit(ClassicWaitingTimesSample)
paramO = cauchy.fit(OrbitWaitingTimesSample)

#%%
X=np.linspace(0,1000,10000)
# Get the pdf from the cauchy distribution
pdf_fittedH = cauchy.pdf(X, *paramH)

# Get the pdf from the cauchy distribution
pdf_fittedC = cauchy.pdf(X, *paramC)

# Get the pdf from the cauchy distribution
pdf_fittedO = cauchy.pdf(X, *paramO)


#%% Plot  pdf of the waiting times in short and long run for the three different parameters
legend_fontsize=30
fig, axs = plt.subplots(1, 2, figsize=(20,7))
plt.subplots_adjust(top=0.8, bottom=0.22,left=0.1, right=0.95)
axs[0].plot(X[0:1000], np.log(pdf_fittedC[0:1000]), color='orange')
axs[0].plot(X[0:1000], np.log(pdf_fittedO[0:1000]), color='blue')
axs[0].plot(X[0:1000], np.log(pdf_fittedH[0:1000]), color='green')
axs[0].plot(X[0:1000], np.log(pdf_fittedC[0:1000]), 's',markevery=100, markersize=7, color='orange', label='Classic')
axs[0].plot(X[0:1000], np.log(pdf_fittedO[0:1000]),'o', markevery=100, markersize=7,color='blue', label='Divergent Oscillator')
axs[0].plot(X[0:1000], np.log(pdf_fittedH[0:1000]),'^',markevery=100, markersize=7, color='green', label='Damped')
axs[0].tick_params(axis='x', labelsize=30)
axs[0].tick_params(axis='y', labelsize=30)
axs[0].set_ylabel('log(density)',  fontsize=30)
axs[0].set_xlabel('number of iterations',fontsize=30)

axs[1].plot(X[1000:10000], np.log(pdf_fittedC[1000:10000]), color='orange')
axs[1].plot(X[1000:10000], np.log(pdf_fittedO[1000:10000]), color='blue' )
axs[1].plot(X[1000:10000], np.log(pdf_fittedH[1000:10000]), color='green')
axs[1].plot(X[1000:10000], np.log(pdf_fittedC[1000:10000]),'s',markevery=1000, markersize=7, color='orange', label='Classic')
axs[1].plot(X[1000:10000], np.log(pdf_fittedO[1000:10000]), 'o', markevery=1000, markersize=7,color='blue', label='Divergent Oscilaltor')
axs[1].plot(X[1000:10000], np.log(pdf_fittedH[1000:10000]),'^',markevery=1000, markersize=7, color='green', label='Damped')
axs[1].tick_params(axis='x', labelsize=30)
axs[1].tick_params(axis='y', labelsize=30)
axs[1].set_xlabel('number of iterations', fontsize=30)
axs[1].set_ylabel('log(density)', fontsize=30)
axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axs[1].set_xlabel('number of iterations', fontsize=30)
axs[1].yaxis.offsetText.set_fontsize(20)
axs[0].legend(bbox_to_anchor=(0.2, 1.02, 1, 0.3), loc="upper left",
               ncol=3,  prop={'size': legend_fontsize})

axs[0].text(0.54, -0.25, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right')
axs[1].text(0.54, -0.25, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right')
plt.savefig('./Plots/Fig10.eps')





# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:36:28 2024

@author: mkost
"""



import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap


# for Fig1 and Fig2: calculates the spectral radius for different parameters and the evolution of the first and second moments

#%%
def fct_abs(c): # complex absolute value
    
    r=c.real
    i=c.imag
    
    return np.sqrt(r**2+i**2)
#%% calculate different operators M and their eigenvalues

Mu=np.linspace(0.01,3.99,100)
W=np.linspace(-1,1.1,100)
WMu=np.zeros((100,100))
WMu1=np.zeros((100,100))
WMuReal=np.zeros((100,100))
WMuImag=np.zeros((100,100))
WMuSign=np.zeros((100,100))
Theta=np.zeros((100,100))

D=np.zeros((100,100))

Convergent=[]
Convergent1=[]

Complex=[]

p=0
for k in range(100):
    w=W[99-k]
    for j in range(100):

        mu=Mu[j]
        M=np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]],
                )
        ev, ew=np.linalg.eig(M)
        
        NormEv=np.zeros(len(ev))
        for i in range(len(ev)):
            NormEv[i]=fct_abs(ev[i])
        
        WMu[k,j]=np.max(NormEv)
        if WMu[k,j]<1:
            Convergent.append([w,mu])
            
        
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        NormEv1=np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i]=fct_abs(ev1[i])
        WMu1[k,j]=np.max(NormEv1)
        if WMu1[k,j]<1:
            Convergent1.append([w,mu])
        
        #if (1+w-mu)**2<4*w:
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        NormEv1=np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i]=fct_abs(ev1[i])
        WMu1[k,j]=np.max(NormEv1)
        if WMu1[k,j]<1:
            
            
            alpha=1+w-mu
            Theta[k,j]=np.abs(np.arctan(np.sqrt(np.abs((1+w-mu)**2-4*w))/alpha))
            D[k,j]=(1+w-mu)**2-4*w
        
        if (1+w-mu)**2<4*w: # check if real
            
            Complex.append([w, mu])
          

#%%



def Divergent():
    
    w=1
    mu=0.03
    
    return w, mu

def Damped():

    mu=0.3215
    w=0.958058
    
    return w, mu

def Overdamped():
    
    w=0.7
    mu=1.4
    
    return w, mu



def C():
    
    w=0.85
    mu=1.4
    
    return w, mu

def A():

    mu=2.2
    w=0.7
    
    return w, mu

def B():
    
    w=0.95
    mu=1
    
    return w, mu


w_divergent, mu_divergent=Divergent()
w_damped, mu_damped=Damped()
w_overdamped, mu_overdamped=Overdamped()

w_A, mu_A=A()
w_B, mu_B=B()
w_C, mu_C=C()

Names=['Divergent O', 'Damped', 'Overdamped', 'Divergent A', 'Divergent B', 'Divergent C']
Markers=['o', '^', 's','v', '1', 'p' ]
Parameters=[[w_divergent, mu_divergent],[w_damped, mu_damped],[w_overdamped, mu_overdamped],[w_A, mu_A],[w_B, mu_B],[w_C, mu_C]]
#%%

# first order stable region

Convergent=np.array(Convergent)
hullF=ConvexHull(Convergent)


Convergent1=np.array(Convergent1)
hullF1=ConvexHull(Convergent1)

Complex=np.array(Complex)
hullF2=ConvexHull(Complex)



#%% get all first order stable pairs 

# Define custom colors for your colormap
colors = [(0.0, 'red'),        # Color for values < 0
          (0.1, 'darkorange'),     # Color transition from 0 to 1
          (0.2, 'yellow'),      # Color transition from 0 to 1
          (0.3, 'lime'),       # Color transition from 1 to 3
          (0.4, 'cyan'),    # Color transition from 1 to 3
          (0.6, 'blue'),
          (0.85, 'blueviolet'),
          (1.0, 'fuchsia')]        # Color for values >=3 

#%% Create the custom colormap
Cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors)
Vmax=2.5
Vmin=0








fig=plt.figure()

plt.imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])


# first order stable region


legend_fontsize=26

for simplex in hullF.simplices:
    plt.plot(Convergent[simplex, 1], Convergent[simplex, 0],linewidth=2, color='grey' )

for simplex in hullF1.simplices:
    plt.plot(Convergent1[simplex, 1], Convergent1[simplex, 0],linewidth=2, color='grey' )

for simplex in hullF2.simplices:
    plt.plot(Complex[simplex, 1], Complex[simplex, 0],linewidth=2, color='grey' )

#plt.title('Spectral radius of B', fontsize=14)

for p in range(len(Parameters)):
    plt.plot(Parameters[p][1], Parameters[p][0], marker=Markers[p],linestyle='None', color='black', label=Names[p])
    #plt.text(Parameters[p][1]+0.1, Parameters[p][0], Names[p], fontsize=12, color='black')

plt.text(1.8,-0.2, 'first-order stability region', fontsize=9, color='black',weight='bold')
plt.text(1.7,0.8, 'second-order stability region', fontsize=9, color='black',weight='bold')
plt.text(0.1,-0.1, 'complex region', fontsize=9, color='black',weight='bold')

# make a box only under the last row  in the centre with latex label \mu
fig.text(0.4, 0.2, r'$\mu$', ha='center', fontsize=16)

# y-axis label next to the left plot
fig.text(0, 0.53, 'w', va='center', rotation='vertical', fontsize=16)
#cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar( shrink=0.82)  # Experiment with the shrink value
plt.legend(bbox_to_anchor=(0, 1.1, 1, 0.3), loc="center",
               ncol=3,  prop={'size': 10}, markerscale=2)

# Increase the size of colorbar ticks
cbar.ax.tick_params(labelsize=16) 
#cbar.ax.tick_params(labelsize=26)
plt.xticks(fontsize=12)  # Adjust the fontsize parameter as needed
plt.yticks(fontsize=12)
#plt.legend(bbox_to_anchor=(-1.41, 2.8, 1, 0.6), loc="upper left",ncol=3,  prop={'size': legend_fontsize})

# Save the figure
plt.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.88)
plt.savefig('Appendix0.png',  bbox_inches="tight")


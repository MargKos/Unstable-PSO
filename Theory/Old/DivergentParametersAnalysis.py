# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:53:59 2024

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

Mu=np.linspace(0.01,0.2,200)
W=np.linspace(0.99, 1.1,200)

WMu=np.zeros((200,200))
WMu1=np.zeros((200,200))


WMuReal=np.zeros((200,200))
WMuImag=np.zeros((200,200))

WMuImagRealPart=np.zeros((200,200))
WMuImagImagPart=np.zeros((200,200))

RealPartSmallerOne=[]
ImagPartSmallerOne=[]
for k in range(200):
    w=W[k]
    for j in range(200):

        mu=Mu[j]
        M=np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]],
                )
        eigenvalues, ew=np.linalg.eig(M)
        
        NormEv=np.zeros(len(eigenvalues))
        for i in range(len(eigenvalues)):
            NormEv[i]=fct_abs(eigenvalues[i])
        
        WMu[k,j]=np.max(NormEv)
 
        # Filter real eigenvalues and get the largest one
        real_eigenvalues = eigenvalues[np.isreal(eigenvalues)]
        
        
        largest_real_eigenvalue = np.max(real_eigenvalues)
        
        
        WMuReal[k,j]=largest_real_eigenvalue
        
        complex_eigenvalues = eigenvalues[np.iscomplex(eigenvalues)]
        
        if np.min(real_eigenvalues)<0 and len(complex_eigenvalues)==2:
            print(real_eigenvalues)
            
        if  len(complex_eigenvalues)==3:
            print(w,mu,eigenvalues)
            
        if len(complex_eigenvalues)>1:
            realpart=complex_eigenvalues[0].real
            imagpart=np.abs(complex_eigenvalues[0].imag)
        
       
            WMuImagRealPart[k,j]=realpart
            WMuImagImagPart[k,j]=imagpart
        
        
        if largest_real_eigenvalue<1:
            RealPartSmallerOne.append([w,mu])
        
      
        
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        NormEv1=np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i]=fct_abs(ev1[i])
        WMu1[k,j]=np.max(NormEv1)

        
#%%


#%% Convex Hull

Hull=np.array(RealPartSmallerOne)
hullF=ConvexHull(Hull)


#%%  

fig=plt.figure()

plt.imshow(WMuImag,  interpolation='nearest',cmap='jet', vmin=0.9, vmax=1, extent=[0.01, 0.99, 0.9, 1.1])

for simplex in hullF.simplices:
    plt.plot(Hull[simplex, 1], Hull[simplex, 0],linewidth=2, color='black' )

plt.colorbar()

#%% Relation between mu and largest real eigenvalue

fig=plt.figure(figsize=(15,10))
plt.plot(Mu, WMu[0, :], '-', linewidth=5, label='spectral radius of B')
plt.plot(Mu, WMuReal[0, :],'--',linewidth=5, label='biggest real eigenvalue of B')
#plt.plot(Mu, WMuImagImagPart[0, :],'--',linewidth=5, label='abs(imag) of complex ev of B')
#plt.plot(Mu, WMuImagRealPart[0, :], '--',linewidth=5,label='real of complex ev of B')
plt.title('w =1', fontsize=30)
plt.xlabel('mu',fontsize=30)
plt.plot(Mu,np.ones(200))
plt.ylabel('value',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig('W=1.png')

#%% Relation between mu and largest real eigenvalue
fig=plt.figure(figsize=(15,10))
plt.plot(W, WMu[:, -1], '-', linewidth=5, label='spectral radius of B')
plt.plot(W, WMuReal[:, -1],'--',linewidth=5, label='biggest real eigenvalue of B')
#plt.plot(Mu, WMuImagImagPart[0, :],'--',linewidth=5, label='abs(imag) of complex ev of B')
#plt.plot(Mu, WMuImagRealPart[0, :], '--',linewidth=5,label='real of complex ev of B')
plt.title('mu =0.2', fontsize=30)
plt.xlabel('w',fontsize=30)
plt.plot(W,np.ones(200))
plt.ylabel('value',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
#plt.savefig('W=1.png')

#%% Relation between mu and largest real eigenvalue
'''
fig=plt.figure(figsize=(15,10))
plt.plot(Mu, WMu[-1, :], '-', linewidth=10, label='spectral radius of B')
plt.plot(Mu, WMuReal[-1, :],'--',linewidth=10, label='biggest real eigenvalue of B')
plt.plot(Mu, WMuImagImagPart[-1, :],'--',linewidth=10, label='abs(imag) of complex ev of B')
plt.plot(Mu, WMuImagRealPart[-1, :], '--',linewidth=10,label='real of complex ev of B')
plt.plot(Mu,np.ones(200))
plt.title('w = 0.958', fontsize=30)
plt.xlabel('mu',fontsize=30)
plt.ylabel('value',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig('W=0.958.png')
'''
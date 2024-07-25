#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:09:20 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt



# for Fig1 and Fig2: calculates the spectral radius for different parameters and the evolution of the first and second moments

#%%
def fct_abs(c): # complex absolute value
    
    r=c.real
    i=c.imag
    
    return np.sqrt(r**2+i**2)
#%% calculate different operators M and their eigenvalues

def calculate_spectral_radi(w, mu):



        
    M=np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]],
                )
    eigenvalues, ew=np.linalg.eig(M)
    
    NormEv=np.zeros(len(eigenvalues))
    for i in range(len(eigenvalues)):
        NormEv[i]=fct_abs(eigenvalues[i])
    
   
 
    # Filter real eigenvalues and get the largest one
    real_eigenvalues = eigenvalues[np.isreal(eigenvalues)]
        
        
    largest_real_eigenvalue = np.max(real_eigenvalues)
        
        
        
        
    complex_eigenvalues = eigenvalues[np.iscomplex(eigenvalues)]
        
    
            
    if len(complex_eigenvalues)>1:
        realpart=complex_eigenvalues[0].real
        imagpart=np.abs(complex_eigenvalues[0].imag)
    
    else:
        realpart=0
        imagpart=0

    M1=np.array([[1+w-mu,-w],[1,0]])
    ev1, ew1=np.linalg.eig(M1)
    NormEv1=np.zeros(len(ev1))
    for i in range(len(ev1)):
        NormEv1[i]=fct_abs(ev1[i])
    
    
    return np.max(NormEv1), np.max(NormEv), largest_real_eigenvalue, realpart, imagpart

#%%

ListA=[]
ListB=[]
W=np.linspace(0.98, 1.01,1000)
for w in W:
    A,B, largest_real_eigenvalue, realpart, imagpart=calculate_spectral_radi(w, 0.14)
    ListA.append(A)
    ListB.append(B)
    
fig=plt.figure()
plt.plot(W, ListB)
plt.plot(W, np.ones(1000))


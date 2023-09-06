#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:34:08 2023

@author: bzfkostr
"""

import numpy as np
from Variables import *


number=8 # number of test functions

X1=[-100,-5.12, -30, -20, -100,0, -400, -600] # lower range
X2=[100,5.12, 30,20, 100, 10, 400, 600]       # upper range

Bdl=[-100,-5.12, -30, -20, -100,0, -100, -600]  # lower boundary
Bdu=[-75,-3.12, -15,-15, -75, 2.5, 100, -450]   # upper boundary

# generate starting points located at boundries for each 1-D test function

def starting_points():  
    StartWerte=np.empty([ dim,n, sim, number])
    for s in range(sim):
        for k in range(number):
            for i in range(n): 
                candidate=np.random.uniform(Bdl[k], Bdu[k])
                StartWerte[:,i, s,k]=candidate
    
    np.save('Boundary1D.npy', StartWerte)
    return StartWerte


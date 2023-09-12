#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:50:25 2023

@author: bzfkostr
"""

import numpy as np
from Variables import *

# define Rastrigin function

def fct_Rastrigin(x):
    
    y = 0
    for i in range(dim):
        if -5.12<x[i]<5.12:
            y=y+x[i]**2-10*np.cos(2*np.pi*x[i])
        else: 
            return (5.12**2+10)*dim+10*dim
    return 10*dim+y


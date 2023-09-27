#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:23:57 2023

@author: bzfkostr
"""

''' Checklist

Set in Variable.py:

- number of particles n
- number of simulations in -sh file
- number of iteration-steps T_PSO
- where to save the files (path)
- select starting points (eventually generate startings points in StartingPoints.py )

1) Run simulations (in -sh file)
1.1 Run PSO simulations PSO5D.py with multiprocessing
1.2 Run ExplorationMultiProcessing5D.py with multiprocessing to calculate for each simulation the number of found local minima
1.3 Run Exploration5D.py to average exploration over all simulations calculated in 1.2, and calculated the commulative number of found local minima 
1.4 Run Measures5D.py to calculate  mean function values and calculate exploitation 

2) Generate Figures

'''

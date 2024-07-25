import numpy as np

# boundaries for one dimensional functions
BoundaryYUp=[10000.0,
 40.352084471444044,
 19.950424956466673,
 400.0,
 100.0,
 5.994994994994995,
 365.17810029105107,
 91.99902347883291]

def f1(x): # smooth (spherical)
    
    return x**2/BoundaryYUp[0]


def f2(x): # rugged Rastrigin
    
    #return 5+2*x**2-2*np.cos(3*np.pi*x)+2
    return (x**2-10*np.cos(2*np.pi*x)+10)/BoundaryYUp[1]

def f3(x): #neutral: Ackley smoothed out

    return (-20*np.exp(-0.2*np.sqrt(x**2))+20)/BoundaryYUp[2]
    
def roundDown(n):
    if n>=0:
        return int(n)
    else:
        return int(n)-1
   
def f4(x): # step-function
    
    return ((roundDown(x+0.5))**2)/BoundaryYUp[3]


def f5(x): # absolute valus
    
    return (np.abs(x))/BoundaryYUp[4]
    
def f6(x): # Hole in the Mountain

    if 0<=x<5:
    
        return (x+1)/BoundaryYUp[5]
    
    if 5<=x<=6:
        return 0
    
    if 6<x<=11:
        
        return (-x+12)/BoundaryYUp[5]
    
    else:
        
        return 1/BoundaryYUp[5]
    

    
def f7(x): # Schwefel 2.26
    if -400<x<350:
    
    
        return (-x*np.sin(np.sqrt(np.abs(x)))+300.5382762038554)/(BoundaryYUp[6]+300.5382762038554)
    
    else :
        
        return (np.abs(-x*np.sin(np.sqrt(np.abs(x))))+300.5382762038554)/(BoundaryYUp[6]+300.5382762038554)


    
def f8(x): # Griewank
    
    
    return (1/4000*x**2-np.cos(x)+1)/BoundaryYUp[7]


def fct_michalewicz(x): # very neutral
    
    m=1
    dim = 5
    y=0
    
    for j in range(dim):
        
        if  0<=x[0]<=np.pi and 0<=x[1]<=np.pi and 0<=x[2]<=np.pi and 0<=x[3]<=np.pi and 0<=x[4]<=np.pi:
    
            y = y +np.sin(x[j])*(np.sin(((j+1)*x[j]**2)/(np.pi)))**(2*m)
        
        else:
            return 0
        
    
    return -y


FunctionList=[f1, f2, f3, f4, f5,f6,f7, f8, fct_michalewicz]
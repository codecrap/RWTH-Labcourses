# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 16:26:42 2017

@author: Daniel
"""

####Güte bererechnunug
import numpy as np
#Gruppe1
R471=46.69
sR471=0.12
R1001=99.45
sR1001=0.25
L1=1.301*10**(-3)
sL1=0.004*10**(-3)
RL1=0.745
sRL1=0.002
C1=4.735*10**(-6)
sC1=0.012*10**(-6)
#Gruppe2
R472=46.55
sR472=0.12
R1002=99.25
sR1002=0.25
L2=4.776*10**(-3)
sL2=0.012*10**(-3)
RL2=3.855
sRL2=0.01
C2=4.719*10**(-6)
sC2=0.012*10**(-6)

def Guete(R,L,RL,C,sR,sL,sRL,sC):
    Q=(R*np.sqrt(C/L))/(1+R*RL*(C/L))
    sQ=np.sqrt(((L**2*np.sqrt(C/L))/(C*R*RL+L)**2)**2*sR**2+((R*(L-C*R*RL))/(2*np.sqrt(C/L)*(C*R*RL+L)**2))**2*sC**2+sRL**2*((C*L*R**2*np.sqrt(C/L))/(C*R*RL+L)**2)**2+sL**2*((R*np.sqrt(C/L)*(C*R*RL-L))/(2*(C*R*RL+L)**2))**2)
    return (Q,sQ)
#Gruppe1
Q471,sQ471 = Guete(R471,L1,RL1,C1,sR471,sL1,sRL1,sC1)
Q1001,sQ1001 = Guete(R1001,L1,RL1,C1,sR1001,sL1,sRL1,sC1)
#Gruppe2
Q472,sQ472=Guete(R472,L2,RL2,C2,sR472,sL2,sRL2,sC2)
Q1002,sQ1002=Guete(R1002,L2,RL2,C2,sR1002,sL2,sRL2,sC2)
'''
Infinity
'''

def Guetinity(RL,L,C,sRL,sL,sC):
    Q=np.sqrt(L/C)/RL
    sQ=np.sqrt(sRL**2*(np.sqrt(L/C)/RL**2)**2+sL**2*(np.sqrt(L*C)/(2*L*RL))**2+sC**2*(np.sqrt(L/C)/(2*C*RL))**2)
    return Q, sQ
Q01, sQ01=Guetinity(RL1,L1,C1,sRL1,sL1,sC1)
Q02, sQ02=Guetinity(RL2,L2,C2,sRL2,sL2,sC2)


print('Gruppe 1')
print('Guete47=%.2f +/- %.9f' % (Q471,sQ471))
print('Guete100=%.2f +/- %.9f' % (Q1001,sQ1001))
print('Guete000=%.2f +/- %.9f' % (Q01,sQ01))

print('Gruppe 2')
print('Guete47=%.2f +/- %.9f' % (Q472,sQ472))
print('Guete100=%.2f +/- %.9f' % (Q1002,sQ1002))
print('Guete000=%.2f +/- %.9f' % (Q02,sQ02))



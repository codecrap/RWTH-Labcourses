# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 20:09:35 2017

@author: Daniel
"""
import numpy as np
import Praktikum as p


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


print('Gruppe 1 Theo')
print('Guete47=%.2f +/- %.9f' % (Q471,sQ471))
print('Guete100=%.2f +/- %.9f' % (Q1001,sQ1001))
print('Guete000=%.2f +/- %.9f' % (Q01,sQ01))

print('Gruppe 2 Theo')
print('Guete47=%.2f +/- %.9f' % (Q472,sQ472))
print('Guete100=%.2f +/- %.9f' % (Q1002,sQ1002))
print('Guete000=%.2f +/- %.9f' % (Q02,sQ02))


#grupe2
Q47=np.array([1.2202,1.245])
eQ47=np.array([0.2,0.024])

Q100=np.array([2.174,2.148])
eQ100=np.array([0.5,0.03])

Q00=np.array([5.96,5.182])
eQ00=np.array([2.6,0.388])

#gruppe1
Q475=np.array([2.03,1.94])
eQ475=np.array([0.22,0.3])

Q1005=np.array([2.95,2.15])
eQ1005=np.array([0.5,0.04])

Q005=np.array([4.21,3.108])
eQ005=np.array([1,0.074])

nQ47, enQ47=p.gewichtetes_mittel(Q47,eQ47)
nQ100, enQ100=p.gewichtetes_mittel(Q100,eQ100)
nQ00, enQ00=p.gewichtetes_mittel(Q00,eQ00)

nQ475, enQ475=p.gewichtetes_mittel(Q475,eQ475)
nQ1005, enQ1005=p.gewichtetes_mittel(Q1005,eQ1005)
nQ005, enQ005=p.gewichtetes_mittel(Q005,eQ005)

print('Gruppe 2 gemittelt')
print('Q47=%.2f +/- %.9f' % (nQ47,enQ47))
print('Q100=%.2f +/- %.9f' % (nQ100,enQ100))
print('Q00=%.2f +/- %.9f' % (nQ00,enQ00))

print('Gruppe 1gemittelt')
print('Q47=%.2f +/- %.9f' % (nQ475,enQ475))
print('Q100=%.2f +/- %.9f' % (nQ1005,enQ1005))
print('Q00=%.2f +/- %.9f' % (nQ005,enQ005))

def delta(x,ex,y,ey):
    delta=abs(x-y)/(np.sqrt(ex**2+ey**2))
    return delta
###2
w247=delta(nQ47,enQ47,Q472,sQ472)
w2100=delta(nQ100,enQ100,Q1002,sQ1002)
w200=delta(nQ00,enQ00,Q02,sQ02)
###1
w147=delta(nQ475,enQ475,Q471,sQ471)
w1100=delta(nQ1005,enQ1005,Q1001,sQ1001)
w100=delta(nQ005,enQ005,Q01,sQ01)

print('Gruppe2')
print(w247)
print(w2100)
print(w200)

print('Gruppe1')
print(w147)
print(w1100)
print(w100)


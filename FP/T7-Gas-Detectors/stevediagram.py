# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:00:57 2019

@author: assil
"""

import scipy
import scipy.fftpack
import scipy.odr
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt,sin,cos,log,exp,diag
import uncertainties.unumpy as unp
from uncertainties import ufloat

#Steverdiagram
Tdmin=[204,212,224]
Tdmax=[232,232,240]
Trmin=[444,472,516]
Trmax=[544,516,540]
Td1=[204,232]
Td2=[212,232]
Td3=[224,240]
Tr1=[444,544]
Tr2=[472,516]
Tr3=[516,540]
"""
a=ufloat(np.mean(Td1),np.std(Td1)/sqrt(2))
print(a)
b=ufloat(np.mean(Td2),np.std(Td2)/sqrt(2))
print(b)
c=ufloat(np.mean(Td3),np.std(Td3)/sqrt(2))
print(c)
d=ufloat(np.mean(Tr1),np.std(Tr1)/sqrt(2))
print(d)
e=ufloat(np.mean(Tr2),np.std(Tr2)/sqrt(2))
print(e)
f=ufloat(np.mean(Tr3),np.std(Tr3)/sqrt(2))
print(f)
"""

Td=[218,222,232]
errTd=[10,7,6]
Tr=[494,494,528]
errTr=[35,16,8]


x=np.mean(Td)
y=np.mean((errTd)/sqrt(3))
T1=ufloat(x,y)
x=np.mean(Tr)
y=np.mean((errTr)/sqrt(3))
T2=ufloat(x,y)

print(T1)
print(T2)
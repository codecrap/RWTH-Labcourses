#-*- coding: utf-8 -*-
#
#@diode_yield.py:
#@author: Olexiy Fedorets
#@date: Tue 19.03.2019


import matplotlib.pyplot as plt
import numpy as np

v0 = []
v1 = []
v01 = []

vI = range(0,50,5)

for i in range(2,12,1):
 vData = np.genfromtxt('Data/Diode-YieldCurve/Diode_'+str(i), skip_header=1)
 v0 += [vData[90][1]]
 v1 += [vData[90][2]]
 v01 += [vData[90][3]]
 
plt.plot(vI,v0)
plt.plot(vI,v1)
plt.plot(vI,v01)